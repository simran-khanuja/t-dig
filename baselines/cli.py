"""All expensive actions require --execute and explicit generation budgets."""

import argparse
import json
from pathlib import Path
from hpdv2.storage import read_json
from benchmarks.coco import prepare_coco
from .config import t2i_model_defaults, T2I_MODELS, validate
from .assets import lock_assets
from .runner import generate


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("config")
    c.add_argument("--method", choices=["iid", "pag", "cads", "group"], required=True)
    c.add_argument("--t2i-model", choices=sorted(T2I_MODELS), default="sd35-large")
    co = sub.add_parser("prepare-coco")
    co.add_argument("--source", type=Path, required=True)
    co.add_argument("--out", type=Path, required=True)
    co.add_argument("--seed", type=int, default=42)
    co.add_argument("--execute", action="store_true")
    a = sub.add_parser("lock-assets")
    a.add_argument("--paths", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--execute", action="store_true")
    r = sub.add_parser("generate")
    r.add_argument("--bundle", type=Path, required=True)
    r.add_argument("--run", type=Path, required=True)
    r.add_argument("--config", type=Path, required=True)
    r.add_argument("--assets", type=Path, required=True)
    r.add_argument("--limit", type=int)
    r.add_argument("--execute", action="store_true")
    r.add_argument("--max-new-images", type=int)
    r.add_argument("--indices", type=int, nargs="+")
    m = sub.add_parser("metrics")
    m.add_argument("--run", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--assets", type=Path, required=True)
    m.add_argument("--execute", action="store_true")
    m.add_argument("--allow-partial", action="store_true")
    m.add_argument("--device", default="cuda:0")
    args = p.parse_args(argv)
    try:
        if args.command == "config":
            result = t2i_model_defaults(args.method, args.t2i_model)
        elif args.command == "prepare-coco":
            result = prepare_coco(args.source, args.out, args.seed, args.execute)
        elif args.command == "lock-assets":
            info = read_json(args.paths)
            result = (
                lock_assets(info["paths"], args.out, info.get("provenance"))
                if args.execute
                else {"execute": False, **info}
            )
        elif args.command == "generate":
            result = generate(
                args.bundle,
                args.run,
                validate(read_json(args.config)),
                args.assets,
                execute=args.execute,
                max_new_images=args.max_new_images,
                limit=args.limit,
                indices=args.indices,
            )
        elif args.command == "metrics":
            from .metrics import evaluate

            result = evaluate(
                args.run,
                args.out,
                args.assets,
                args.execute,
                args.allow_partial,
                args.device,
            )
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError) as error:
        p.exit(2, f"error: {error}\n")
