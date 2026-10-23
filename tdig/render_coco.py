"""Render the COCO-250 captions with each baseline method of one T2I model.

Reads the subset written by `tdig.coco_fid select` and, for every method configuration given,
renders its bundle indices with `baselines generate` against the COCO bundle the subset was drawn
from. Configurations are the files `baselines config --method <m> --t2i-model <r>` writes, one per
method. Seeds derive from the configuration seed, the prompt id and the image slot, so the images
do not depend on the GPU split.

Each method's indices are split over the node's GPUs into run directories
`<out>/<method>-gpu<N>`, in the `baselines.runner` layout that `tdig.coco_fid index` reads.
Rerunning with the same `--out` continues: `baselines generate` reuses verified images.
"""

import argparse
import json
import os
import subprocess
import threading
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--subset", type=Path, required=True, help="subset.json written by tdig.coco_fid select"
    )
    p.add_argument(
        "--config",
        type=Path,
        action="append",
        required=True,
        help="method configuration from `baselines config`; repeat once per method",
    )
    p.add_argument("--assets", type=Path, required=True, help="asset lock of the T2I model")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--gpus", type=int, help="GPUs to split over (default: every GPU this process may see)"
    )
    p.add_argument(
        "--python",
        default=os.environ.get("PY_REWARD", "python"),
        help="T2I model interpreter (default: $PY_REWARD, else python)",
    )
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    # `gpu` indexes the visible GPUs: with CUDA_VISIBLE_DEVICES=6,7, worker 0 gets device 6.
    visible = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x]
    if not visible:
        listed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        visible = [x.strip() for x in listed]
    a.gpus = a.gpus or len(visible)
    if not 0 < a.gpus <= len(visible):
        raise SystemExit(f"--gpus {a.gpus} but {len(visible)} GPUs are visible")
    indices = [r["index"] for r in json.loads(a.subset.read_text())["rows"]]
    jobs = []  # (gpu, method, config path, k, indices)
    for path in a.config:
        config = json.loads(path.read_text())
        for gpu in range(a.gpus):
            part = indices[gpu :: a.gpus]
            if part:
                jobs.append((gpu, config["method"], path.resolve(), config["k"], part))
    if len({j[1] for j in jobs}) != len(a.config):
        raise SystemExit("two configurations share a method")
    print(
        json.dumps(
            dict(
                captions=len(indices),
                runs=len(jobs),
                images=sum(j[3] * len(j[4]) for j in jobs),
                methods=sorted({j[1] for j in jobs}),
            ),
            indent=2,
        ),
        flush=True,
    )
    if not a.execute:
        return
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    failures = []

    def worker(gpu):
        for _, method, config, k, part in [j for j in jobs if j[0] == gpu]:
            run = out / f"{method}-gpu{gpu}"
            command = [
                a.python,
                "-u",
                "-m",
                "baselines",
                "generate",
                "--bundle",
                str(a.bundle.resolve()),
                "--run",
                str(run),
                "--config",
                str(config),
                "--assets",
                str(a.assets.resolve()),
                "--indices",
                *map(str, part),
                "--execute",
                "--max-new-images",
                str(k * len(part)),
            ]
            env = os.environ.copy()
            env.update(
                CUDA_VISIBLE_DEVICES=visible[gpu],
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                PYTHONUNBUFFERED="1",
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
                PYTHONPATH=str(root),
            )
            with (out / f"{method}-gpu{gpu}.log").open("a") as stream:
                code = subprocess.run(
                    command,
                    cwd=root,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                ).returncode
            print(f"{method}-gpu{gpu} exit={code}", flush=True)
            if code:
                failures.append(f"{method}-gpu{gpu}")

    threads = [threading.Thread(target=worker, args=(g,)) for g in range(a.gpus)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failures:
        raise SystemExit(f"failed: {failures}")


if __name__ == "__main__":
    main()
