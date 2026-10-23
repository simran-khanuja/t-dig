"""Explicit prepare/filter/cluster commands; paid work is opt-in."""

import argparse
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

from .api import LiteLLMClient
from .cluster import cluster
from .dataset import load_input, prepare
from .runner import execute, make_manifest, plan
from .review import build_review
from .policy import system_prompt


def integer(minimum):
    def parse(text):
        value = int(text)
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be >= {minimum}")
        return value

    return parse


def positive_float(text):
    value = float(text)
    import math

    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def commit(value):
    if not re.fullmatch(r"[a-fA-F0-9]{40}", value):
        raise argparse.ArgumentTypeError("use a full immutable 40-character commit hash")
    return value


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    policy = sub.add_parser("show-policy", help="Print the exact fully assembled system prompt")
    policy.add_argument("--stage", choices=["pass1", "pass2"], required=True)
    prep = sub.add_parser("prepare", help="Create a versioned pool from HPDv2 JSON")
    source = prep.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", type=Path, help="Local train.json or test.json")
    source.add_argument(
        "--download", action="store_true", help="Explicitly permit HF dataset download"
    )
    prep.add_argument("--revision", type=commit)
    prep.add_argument("--split", choices=["train", "test"], required=True)
    prep.add_argument("--output", type=Path, required=True)

    filt = sub.add_parser("filter", help="Plan or execute pass1/pass2")
    filt.add_argument("--stage", choices=["pass1", "pass2"], required=True)
    filt.add_argument("--input", type=Path, required=True)
    filt.add_argument("--run-dir", type=Path, required=True)
    filt.add_argument("--backend", choices=["proxy", "anthropic"], required=True)
    filt.add_argument("--api-base", help="Proxy base URL; no implicit institutional endpoint")
    filt.add_argument("--api-key-env", help="Environment variable name; never the key itself")
    filt.add_argument(
        "--model", required=True, help="Exact proxy model alias or Anthropic model ID"
    )
    filt.add_argument(
        "--short-request-ids",
        action="store_true",
        help="Use exact batch-local IDs on the wire; preserve full hashes in saved decisions",
    )
    filt.add_argument("--batch-size", type=integer(1), default=20)
    filt.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
        help="Explicit reasoning effort for compatible proxy models; omits temperature",
    )
    filt.add_argument("--max-tokens", type=integer(1), default=8192)
    filt.add_argument("--timeout", type=positive_float, default=120)
    filt.add_argument(
        "--cache-system", action="store_true", help="Request Anthropic-style system caching"
    )
    filt.add_argument("--workers", type=integer(1), default=1)
    filt.add_argument(
        "--attempts", type=integer(1), default=3, help="Total attempts per batch, including first"
    )
    filt.add_argument(
        "--limit",
        type=integer(0),
        help="Pending prompts to attempt, including prior errors; 0 means none",
    )
    filt.add_argument(
        "--max-requests",
        type=integer(1),
        help="Per-invocation cap INCLUDING retry calls; required for --execute",
    )
    filt.add_argument("--execute", action="store_true", help="Permit paid API calls and run writes")
    filt.add_argument(
        "--repair-tail",
        action="store_true",
        help="Back up and truncate an incomplete final journal line",
    )
    filt.add_argument(
        "--allow-partial-input",
        action="store_true",
        help="Pilot only: accept incomplete pass1 output",
    )

    cl = sub.add_parser("cluster", help="Plan or execute global clustering")
    cl.add_argument("--input", type=Path, required=True)
    cl.add_argument("--output", type=Path, required=True)
    cl.add_argument("--distance", type=positive_float, required=True)
    cl.add_argument("--max-per-cluster", type=integer(1), default=1)
    cl.add_argument("--selection", choices=["random", "max-min"], default="random")
    cl.add_argument(
        "--seed", type=int, default=42, help="Reproducible random selection within clusters"
    )
    cl.add_argument("--min-cluster-size", type=integer(1), default=1)
    cl.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    cl.add_argument("--embedding-revision", type=commit)
    cl.add_argument("--device", default="cpu")
    cl.add_argument("--max-matrix-gib", type=positive_float, default=4)
    cl.add_argument("--execute", action="store_true")
    cl.add_argument("--allow-download", action="store_true", help="Permit embedding model download")
    cl.add_argument("--allow-partial-input", action="store_true", help="Pilot only")
    review = sub.add_parser("review", help="Write a standalone stratified manual-review page")
    review.add_argument("--input", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--per-group", type=integer(1), default=20)
    review.add_argument("--seed", type=int, default=42)
    return ap


def run(args):
    if args.command == "show-policy":
        print(system_prompt(args.stage))
        return 0
    if args.command == "review":
        print(
            json.dumps(build_review(args.input, args.output, args.per_group, args.seed), indent=2)
        )
        return 0
    if args.command == "prepare":
        if args.output.exists():
            raise ValueError("Output exists; choose a new pool path")
        provenance = None
        if args.download:
            if not args.revision:
                raise ValueError("--download requires --revision")
            from huggingface_hub import hf_hub_download

            args.source = hf_hub_download(
                "ymhao/HPDv2",
                filename=f"{args.split}.json",
                repo_type="dataset",
                revision=args.revision,
            )
            provenance = {"repo": "ymhao/HPDv2", "revision": args.revision}
        elif args.revision:
            raise ValueError("--revision applies to --download; local sources are content-hashed")
        result = prepare(args.source, args.output, args.split, provenance)
        print(json.dumps(result["metadata"], indent=2))
        return 0
    if args.command == "cluster":
        result = cluster(
            args.input,
            args.output,
            distance=args.distance,
            cap=args.max_per_cluster,
            selection=args.selection,
            seed=args.seed,
            min_size=args.min_cluster_size,
            model=args.embedding_model,
            revision=args.embedding_revision,
            device=args.device,
            max_matrix_gib=args.max_matrix_gib,
            execute=args.execute,
            allow_download=args.allow_download,
            allow_partial=args.allow_partial_input,
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.backend == "proxy":
        url = urlparse(args.api_base or "")
        if (
            url.scheme not in {"http", "https"}
            or not url.netloc
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("Proxy requires --api-base URL without credentials, query or fragment")
        if args.model.startswith("litellm_proxy/"):
            raise ValueError("Pass the proxy's model alias without the litellm_proxy/ prefix")
    elif args.api_base:
        raise ValueError("--api-base is only allowed with --backend proxy")
    if args.reasoning_effort is not None and (args.backend != "proxy" or args.cache_system):
        raise ValueError(
            "--reasoning-effort requires proxy backend without Anthropic-style caching"
        )
    if args.repair_tail and not args.execute:
        raise ValueError("--repair-tail requires --execute")
    if args.execute and args.max_requests is None:
        raise ValueError("--execute requires --max-requests (includes retries)")
    data, rows = load_input(args.input, args.stage, args.allow_partial_input)
    config = {
        k: getattr(args, k)
        for k in (
            "stage",
            "backend",
            "api_base",
            "model",
            "batch_size",
            "max_tokens",
            "timeout",
            "cache_system",
            "allow_partial_input",
            "reasoning_effort",
            "short_request_ids",
        )
    }
    manifest = make_manifest(data, rows, config)
    if not args.execute:
        print(
            json.dumps(
                {
                    **plan(args.run_dir, rows, manifest, args.limit),
                    "execute": False,
                    "max_requests": args.max_requests,
                    "model": args.model,
                    "backend": args.backend,
                    "api_base": args.api_base,
                },
                indent=2,
            )
        )
        return 0
    key_env = args.api_key_env or (
        "LITELLM_API_KEY" if args.backend == "proxy" else "ANTHROPIC_API_KEY"
    )
    key = os.environ.get(key_env)
    # limit=0 permits rebuilding a summary without a key or API dependency.
    client = None
    if args.limit != 0:
        if not key:
            raise ValueError(f"Missing API key environment variable: {key_env}")
        client = LiteLLMClient(config, key)
    result = execute(
        args.run_dir,
        rows,
        manifest,
        client,
        limit=args.limit,
        max_requests=args.max_requests,
        workers=args.workers,
        attempts=args.attempts,
        repair_tail=args.repair_tail,
    )
    print(json.dumps(result["metadata"], indent=2))
    return 0 if result["metadata"]["complete"] else 2


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
