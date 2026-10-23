import argparse
import json
import os

from hpdv2.storage import atomic_json, read_json
from .selection import prepare
from .examples import inventory
from .runner import run


def main():
    parser = argparse.ArgumentParser(
        description='Balanced, image-conditioned paired axis elicitation'
    )
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--render-root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--per-t2i-model', type=int, default=5000)
    p.add_argument('--seed', type=int, default=42)
    p = commands.add_parser('inventory')
    p.add_argument('--human-study', required=True)
    p.add_argument('--study-run', required=True)
    p.add_argument('--out', required=True)
    p = commands.add_parser('run')
    for name in ('assignment', 'examples', 'config', 'out'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--limit', type=int, default=40)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--max-calls', type=int)
    p.add_argument('--api-key-env', default='LITELLM_API_KEY')
    p.add_argument('--retry-errors', action='store_true')
    p.add_argument('--retry-uncertain', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.render_root, args.out, args.per_t2i_model, args.seed)
        print(json.dumps({'selected': len(result['items']), 'per_renderer': args.per_t2i_model}))
    elif args.command == 'inventory':
        result = inventory(args.human_study, args.study_run)
        atomic_json(args.out, result)
        print(json.dumps({'counts': result['counts'], 'excluded': result['excluded']}))
    else:
        result = run(
            args.assignment,
            args.examples,
            args.out,
            read_json(args.config),
            limit=args.limit,
            execute=args.execute,
            max_calls=args.max_calls,
            api_key=os.environ.get(args.api_key_env),
            retry_errors=args.retry_errors,
            retry_uncertain=args.retry_uncertain,
        )
        print(json.dumps(result))


if __name__ == '__main__':
    main()
