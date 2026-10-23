"""Export a verl FSDP checkpoint as a PEFT adapter directory vLLM can serve.

verl writes the actor checkpoint as sharded FSDP state dicts
(`global_step_N/actor/model_world_size_2_rank_*.pt`) plus `lora_train_meta.json`, without
`adapter_config.json` unless the run requested it, so `--lora <ckpt dir>` cannot load it
directly. This script extracts the adapter with verl's own `FSDPModelMerger` and
`save_lora_adapter`.

Only the ~130 MB adapter is written; `python -m verl.model_merger merge` would also write the
~8 GB merged base model. Pass `--full` to write the merged model too.

    $PY_TRAIN \\
        axis_framework/export_lora.py \\
        --ckpt runs/axis_grpo/ckpt/k2_vqaalign_lam0 \\
        --step 879
    # -> axis_framework/adapters/k2_vqaalign_lam0_step879/lora_adapter

Then serve it:  policy_server.py --lora axis_framework/adapters/<name>
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from axis_framework import config  # noqa: E402


def resolve_step(ckpt, step):
    """Return (<ckpt>/global_step_N/actor, N), defaulting to the step verl last recorded.

    The default reads `latest_checkpointed_iteration.txt`, which verl updates only after a
    write completes; a `global_step_N` directory can be half-written after a preemption.
    The highest `global_step_N` is the fallback when that file is missing.

    Raises:
        FileNotFoundError: no `global_step_N` under *ckpt*, or the actor dir is missing.
    """
    if step is None:
        latest = os.path.join(ckpt, "latest_checkpointed_iteration.txt")
        if os.path.exists(latest):
            step = open(latest).read().strip()
        else:
            steps = [
                int(d.split("_")[-1])
                for d in os.listdir(ckpt)
                if re.fullmatch(r"global_step_\d+", d)
            ]
            if not steps:
                raise FileNotFoundError(f"no global_step_N under {ckpt}")
            step = max(steps)
    d = os.path.join(ckpt, f"global_step_{step}", "actor")
    if not os.path.isdir(d):
        raise FileNotFoundError(f"{d} does not exist")
    return d, int(step)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--ckpt",
        required=True,
        help="an experiment dir (axis_grpo/ckpt/<EXP>) or a global_step_N/actor",
    )
    ap.add_argument(
        "--step",
        default=None,
        help="which global_step to export; default = latest_checkpointed_iteration",
    )
    ap.add_argument("--out", default=None, help="default: axis_framework/adapters/<EXP>_step<N>")
    ap.add_argument(
        "--full",
        action="store_true",
        help="also write the merged base model (~8 GB) beside the adapter",
    )
    a = ap.parse_args()

    ckpt = os.path.abspath(os.path.expanduser(a.ckpt))
    if os.path.basename(ckpt) == "actor":
        actor, step = ckpt, -1
        exp = os.path.basename(os.path.dirname(os.path.dirname(ckpt)))
    else:
        actor, step = resolve_step(ckpt, a.step)
        exp = os.path.basename(ckpt)

    out = a.out or os.path.join(config.FRAMEWORK, "adapters", f"{exp}_step{step}")
    os.makedirs(out, exist_ok=True)

    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    cfg = ModelMergerConfig(operation="merge", backend="fsdp", local_dir=actor, target_dir=out)
    # Read the base model config from the checkpoint's own actor/huggingface/, so the
    # adapter matches the architecture the run trained.
    cfg.hf_model_config_path = os.path.join(actor, "huggingface")
    merger = FSDPModelMerger(cfg)

    world_size = merger._get_world_size()
    sd = merger._load_rank_zero_state_dict(world_size)
    mesh, names = merger._extract_device_mesh_info(sd, world_size)
    shards, shape = merger._calculate_shard_configuration(mesh, names)
    print(f"[export] {exp} step {step}: {world_size} ranks, {shards} shards {shape}", flush=True)
    merged = merger._load_and_merge_state_dicts(world_size, shards, shape, names)

    if a.full:
        merger.save_hf_model_and_tokenizer(merged)  # writes the adapter too
        lora_path = os.path.join(out, "lora_adapter")
    else:
        lora_path = merger.save_lora_adapter(merged)
    if not lora_path:
        raise SystemExit(
            f"no LoRA parameters in {actor}. That checkpoint is a FULL fine-tune, not a "
            f"LoRA run -- serve it with --model <dir> instead of --lora."
        )

    n = sum(os.path.getsize(os.path.join(lora_path, f)) for f in os.listdir(lora_path)) / 2**20
    print(f"[export] adapter -> {lora_path}  ({n:.0f} MiB)")
    print(f"[export] serve with: policy_server.py --lora {out}")


if __name__ == "__main__":
    main()
