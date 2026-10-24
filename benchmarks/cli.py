"""Prepare immutable benchmark bundles.

Only the Python standard library is required. Preparation plans by default;
--execute writes a new directory, --allow-download additionally permits network.
"""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()


def registry():
    return json.loads(Path(__file__).with_name('sources.lock.json').read_text())


def make_rows(name, raw, spec):
    if digest(raw) != spec['sha256']:
        raise ValueError('Source SHA256 mismatch; refusing a changed benchmark')
    source_rows = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
    if len(source_rows) != spec['count']:
        raise ValueError('Source row count mismatch')
    rows = []
    for index, item in enumerate(source_rows):
        prompt = item.get('initial_prompt')
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f'Invalid prompt at row {index}')
        rows.append(
            {'id': f'{name}:{index:05d}', 'index': index, 'prompt': prompt, 'metadata': item}
        )
    return rows


def prepare(name, out, source=None, execute=False, allow_download=False):
    spec = registry()[name]
    plan = {
        'benchmark': name,
        'source': spec,
        'output': str(out),
        'execute': execute,
        'count': spec['count'],
    }
    if not execute:
        return plan  # Planning never reads the network or creates files.
    if out.exists():
        raise ValueError(f'Refusing to overwrite {out}')
    if source is not None:
        raw = source.read_bytes()
    elif allow_download:
        with urllib.request.urlopen(spec['url'], timeout=60) as response:
            raw = response.read()
    else:
        raise ValueError('Provide --source or explicitly enable --allow-download')
    rows = make_rows(name, raw, spec)
    row_bytes = ''.join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows
    ).encode()
    manifest = {
        'schema_version': 1,
        'benchmark': name,
        'count': len(rows),
        'source': spec,
        'rows_sha256': digest(row_bytes),
        'selection': 'all released rows, original order, no filtering or deduplication',
    }
    # A manifest is written last: partial preparation is never a valid bundle.
    out.mkdir(parents=True, exist_ok=False)
    (out / 'source.jsonl').write_bytes(raw)
    (out / 'prompts.jsonl').write_bytes(row_bytes)
    (out / 'manifest.json').write_bytes(encode(manifest))
    return manifest


def load_bundle(path):
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported benchmark schema')
    name = manifest['benchmark']
    if manifest['source'] != registry()[name]:
        raise ValueError('Source manifest differs from pinned registry')
    expected = make_rows(name, (path / 'source.jsonl').read_bytes(), manifest['source'])
    raw = (path / 'prompts.jsonl').read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    if digest(raw) != manifest['rows_sha256'] or rows != expected or len(rows) != manifest['count']:
        raise ValueError('Benchmark content mismatch')
    return manifest, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare', help='Plan or prepare a pinned benchmark')
    prep.add_argument('--benchmark', choices=sorted(registry()), required=True)
    prep.add_argument('--out', type=Path, required=True)
    prep.add_argument('--source', type=Path)
    prep.add_argument('--execute', action='store_true')
    prep.add_argument('--allow-download', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = prepare(args.benchmark, args.out, args.source, args.execute, args.allow_download)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f'error: {exc}\n')
