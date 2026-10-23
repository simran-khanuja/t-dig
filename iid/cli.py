"""Plan or execute verified i.i.d. generations from a curated prompt pool."""

import argparse
import json
import math
from pathlib import Path
import sys

from .runner import generate
from .service import RenderService


def positive(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def nonnegative(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError('must be nonnegative')
    return value


def finite_float(text):
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError('must be finite and nonnegative')
    return value


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--input', type=Path, required=True, help='New clustered_pool JSON; axes are not used'
    )
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument(
        '--k',
        type=positive,
        default=4,
        help='Number of images per prompt; any positive integer (default 4)',
    )
    ap.add_argument(
        '--t2i-url',
        required=True,
        help='Existing axis_framework render service; shared filesystem',
    )
    ap.add_argument('--model', required=True, help='Exact served model alias')
    ap.add_argument(
        '--model-path', required=True, help='Exact checkpoint path expected in T2I model responses'
    )
    ap.add_argument(
        '--t2i-model-revision',
        required=True,
        help='Operator-pinned immutable checkpoint revision; recorded, not remotely attested',
    )
    ap.add_argument('--resolution', type=positive, required=True)
    ap.add_argument('--steps', type=positive, required=True)
    ap.add_argument('--guidance', type=finite_float, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--timeout', type=finite_float, default=600)
    ap.add_argument(
        '--limit',
        type=nonnegative,
        help='Maximum pending prompt sets to work on; 0 makes no requests',
    )
    ap.add_argument(
        '--max-requests',
        type=positive,
        help='Maximum image requests this invocation; required with --execute',
    )
    ap.add_argument(
        '--execute',
        action='store_true',
        help='Permit image generation and run writes; otherwise read-only planning',
    )
    ap.add_argument(
        '--allow-partial-input',
        action='store_true',
        help='Pilot only: accept an incomplete upstream curation',
    )
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.timeout <= 0:
            raise ValueError('Timeout must be positive')
        RenderService(args.t2i_url, args.timeout)  # URL validation only, no network.
        if (
            not args.model.strip()
            or not args.model_path.strip()
            or not args.t2i_model_revision.strip()
        ):
            raise ValueError('Model, checkpoint path and revision must be nonempty')
        # The run manifest keeps its stored field names `render_url` and `renderer_revision`.
        cfg = {
            name: getattr(args, name)
            for name in [
                'k',
                'model',
                'model_path',
                'resolution',
                'steps',
                'guidance',
                'seed',
                'timeout',
            ]
        }
        cfg.update(render_url=args.t2i_url, renderer_revision=args.t2i_model_revision)
        result = generate(
            args.input,
            args.run_dir,
            cfg,
            execute=args.execute,
            max_requests=args.max_requests,
            limit=args.limit,
            allow_partial=args.allow_partial_input,
        )
        print(json.dumps(result, indent=2))
        return 0 if not args.execute or result.get('complete') else 2
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
