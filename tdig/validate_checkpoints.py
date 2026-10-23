"""Score saved checkpoints of one training run on held-out prompt sets, outside training.

For each requested optimizer step the script:

1. merges the checkpoint's FSDP shards into a PEFT LoRA adapter with `verl.model_merger`;
2. samples one completion per row with vLLM, under the sampling training uses for validation
   (temperature 1.0, the chat template with `enable_thinking=False`);
3. scores every completion with `axis_grpo.training.reward_fn.compute_score`, the function the
   trainer calls, against reward services running with the training run's reward configuration.

Step 0 is the policy before training: the base model with no adapter.

Rows carry `extra_info.split == "val"`, so the reward service renders them at the validation
settings of the run's `reward-config.json` (1024 px, inference step counts), not at the 512 px
training settings. The services must run with `save_images: true`, as
`tdig/validate_checkpoints.sh` starts them.

Reads `<run>/train-config.json` for the policy snapshot, T2I model, reward, LoRA rank, token
limits and tracked scorers, so a checkpoint is scored under the settings it was trained with.

Run in the trainer environment (PY_TRAIN, requirements/train.txt) on a node with one free GPU for
vLLM and reachable reward services.
"""

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import urllib.request
from pathlib import Path

from axis_grpo.training.protocol import (
    malformed_score,
    parse_prompts,
    parse_spec,
    request_id,
    seed_for,
)


def load_rows(path):
    """Return the rows of one policy-format parquet file as a list of dicts."""
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def checkpoint_steps(run, selection):
    """Return the optimizer steps to score, in ascending order.

    Args:
        run: training run directory; checkpoints are `<run>/policy/checkpoints/global_step_<N>`
            and count only when `tdig-complete.json` exists.
        selection: list of strings. `all`, `latest`, `every:<N>` (multiples of N), or integers.
            `0` selects the untrained base model and needs no checkpoint.

    Raises `ValueError` for a requested step without a complete checkpoint.
    """
    root = Path(run) / "policy" / "checkpoints"
    saved = sorted(
        int(p.name.removeprefix("global_step_"))
        for p in root.glob("global_step_*")
        if (p / "tdig-complete.json").is_file()
    )
    chosen = set()
    for item in selection:
        if item == "all":
            chosen.update(saved)
        elif item == "latest":
            chosen.update(saved[-1:])
        elif item.startswith("every:"):
            chosen.update(s for s in saved if s % int(item.split(":")[1]) == 0)
        else:
            chosen.add(int(item))
    missing = sorted(s for s in chosen if s and s not in saved)
    if missing:
        raise ValueError(f"No complete checkpoint for steps {missing}; saved: {saved}")
    return sorted(chosen)


def merge_adapter(run, step, out):
    """Merge one checkpoint into a PEFT adapter and return the adapter directory.

    Writes `<out>/adapters/step_<N>/lora_adapter/`. A directory that already holds
    `adapter_config.json` is reused. Raises `RuntimeError` when the merger writes no adapter.
    """
    target = Path(out) / "adapters" / f"step_{step}"
    adapter = target / "lora_adapter"
    if not (adapter / "adapter_config.json").is_file():
        actor = Path(run) / "policy" / "checkpoints" / f"global_step_{step}" / "actor"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "verl.model_merger",
                "merge",
                "--backend",
                "fsdp",
                "--local_dir",
                str(actor),
                "--target_dir",
                str(target),
                "--use_cpu_initialization",
            ],
            check=True,
        )
        if not (adapter / "adapter_config.json").is_file():
            raise RuntimeError(f"model_merger wrote no lora_adapter under {target}")
    return adapter


def conversations(rows):
    """Return vLLM chat messages for rows that carry policy input images.

    The n-th `<image>` placeholder of a row's prompt is replaced by the n-th path of its `images`
    column, the pairing verl's dataset uses in training. Raises `ValueError` when the counts differ.
    """
    from PIL import Image

    result = []
    for row in rows:
        paths = iter(row["images"])
        messages = []
        for message in row["prompt"]:
            pieces = message["content"].split("<image>")
            parts = []
            for i, text in enumerate(pieces):
                if text:
                    parts.append(dict(type="text", text=text))
                if i < len(pieces) - 1:
                    parts.append(
                        dict(type="image_pil", image_pil=Image.open(next(paths)).convert("RGB"))
                    )
            messages.append(dict(role=message["role"], content=parts))
        if next(paths, None) is not None:
            raise ValueError("A row has more images than <image> placeholders")
        result.append(messages)
    return result


def score_rows(rows, completions, cfg, endpoints, concurrency):
    """Score completions with the trainer's reward client and return one metric dict per row.

    Each dict also holds `request_id`, under which the reward service stored the request's
    images and per-image scores; `None` for a malformed completion, which is never rendered.
    """
    from axis_grpo.training.reward_fn import compute_score

    routes = {cfg["renderer"]: endpoints}

    async def one(row, text, gate):
        async with gate:
            metrics = await compute_score(
                data_source=row["data_source"],
                solution_str=text,
                ground_truth=row["reward_model"]["ground_truth"],
                extra_info=row["extra_info"],
                routes=routes,
                reward=cfg["reward"],
                renderer=cfg["renderer"],
                seed=cfg["render_seed"],
                timeout=cfg["reward_timeout"],
                tracked=cfg["track_metrics"],
            )
        spec = parse_spec(row["reward_model"]["ground_truth"])
        rid = None
        if not metrics["malformed"]:
            item = dict(
                **spec,
                prompts=parse_prompts(text, spec["k"]),
                seed=seed_for(
                    spec["base_prompt"], spec["tpips_factor"], cfg["render_seed"], "base"
                ),
                split=row["extra_info"].get("split", "train"),
                row_id=row["extra_info"].get("row_id", ""),
            )
            rid = request_id(item)
        return dict(metrics, request_id=rid)

    async def everything():
        gate = asyncio.Semaphore(concurrency)
        return await asyncio.gather(
            *(one(r, t, gate) for r, t in zip(rows, completions, strict=True))
        )

    return asyncio.run(everything())


def summarise(records):
    """Return counts and the mean of every metric over the completions that parsed."""
    valid = [r for r in records if not r["metrics"]["malformed"]]
    names = sorted(k for k in records[0]["metrics"] if k not in ("request_id",))
    words = [len(p.split()) for r in valid for p in parse_prompts(r["completion"], r["k"])]
    return dict(
        rows=len(records),
        malformed=len(records) - len(valid),
        words_per_prompt=statistics.mean(words) if words else None,
        mean_valid=(
            {k: statistics.mean(r["metrics"][k] for r in valid) for k in names} if valid else {}
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--run", type=Path, required=True, help="training run directory holding train-config.json"
    )
    p.add_argument(
        "--set", action="append", required=True, metavar="NAME=PARQUET", help="prompt set to score"
    )
    p.add_argument(
        "--steps",
        nargs="+",
        default=["latest"],
        help="all | latest | every:<N> | integers; 0 = untrained",
    )
    p.add_argument("--endpoint", action="append", required=True, help="reward service URL")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpu-memory-fraction", type=float, default=0.85)
    p.add_argument("--concurrency", type=int, default=48, help="completions scored at once")
    p.add_argument(
        "--limit", type=int, help="score only the first N rows of each set; for a smoke test"
    )
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    cfg = json.loads((a.run / "train-config.json").read_text())
    sets = {}
    for item in a.set:
        name, path = item.split("=", 1)
        sets[name] = load_rows(path)[: a.limit]
    steps = checkpoint_steps(a.run, a.steps)
    plan = dict(
        run=str(a.run.resolve()),
        renderer=cfg["renderer"],
        reward=cfg["reward"],
        steps=steps,
        sets={k: len(v) for k, v in sets.items()},
        policy_observation=cfg["policy_observation"],
        completions=len(steps) * sum(len(v) for v in sets.values()),
    )
    print(json.dumps(plan, indent=2), flush=True)
    if not a.execute:
        return
    for url in a.endpoint:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=30) as response:
            health = json.loads(response.read())
        if (
            health.get("renderer") != cfg["renderer"]
            or health.get("track_metrics", []) != cfg["track_metrics"]
        ):
            raise ValueError(
                f"Reward service {url} does not match the run's T2I model or tracked scorers"
            )
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    pending = [
        s
        for s in steps
        if not all((a.out / f"step_{s}" / f"{n}.summary.json").exists() for n in sets)
    ]
    if not pending:
        print("every requested step is already scored")
        return
    adapters = {s: merge_adapter(a.run, s, a.out) for s in pending if s}

    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    visual = cfg["policy_observation"] == "elicitation_images"
    options = dict(
        model=cfg["policy_path"],
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=cfg["lora_rank"],
        max_model_len=cfg["max_prompt_length"] + cfg["max_response_length"],
        gpu_memory_utilization=a.gpu_memory_fraction,
        max_num_seqs=64,
        seed=cfg["seed"],
    )
    if cfg.get("gdn_prefill_backend"):
        options["gdn_prefill_backend"] = cfg["gdn_prefill_backend"]
    if visual:
        options["limit_mm_per_prompt"] = {"image": cfg["input_images"]}
    llm = LLM(**options)
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        temperature=1.0, max_tokens=cfg["max_response_length"], seed=cfg["seed"]
    )

    for step in pending:
        lora = LoRARequest(f"step-{step}", step, str(adapters[step])) if step else None
        for name, rows in sets.items():
            target = a.out / f"step_{step}"
            if (target / f"{name}.summary.json").exists():
                continue
            if visual:
                outputs = llm.chat(
                    conversations(rows),
                    sampling,
                    lora_request=lora,
                    add_generation_prompt=True,
                    chat_template_kwargs=dict(enable_thinking=False),
                )
            else:
                prompts = [
                    tokenizer.apply_chat_template(
                        r["prompt"],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    for r in rows
                ]
                outputs = llm.generate(prompts, sampling, lora_request=lora)
            completions = [o.outputs[0].text for o in outputs]
            metrics = score_rows(rows, completions, cfg, a.endpoint, a.concurrency)
            records = [
                dict(
                    row_id=r["extra_info"]["row_id"],
                    base_prompt=r["extra_info"]["base_prompt"],
                    axis=r["extra_info"]["axis"],
                    k=r["extra_info"]["k"],
                    completion=c,
                    request_id=m.pop("request_id"),
                    metrics=m,
                )
                for r, c, m in zip(rows, completions, metrics, strict=True)
            ]
            target.mkdir(parents=True, exist_ok=True)
            (target / f"{name}.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
            )
            summary = dict(
                step=step,
                set=name,
                malformed_score=malformed_score(cfg["reward"]),
                **summarise(records),
            )
            (target / f"{name}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
