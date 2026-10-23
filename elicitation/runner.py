"""Bounded elicitation runner: pinned requests, raw replies saved, explicit resume.

One request per prompt, sent serially.
"""

import base64
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from urllib.parse import urlsplit

from hpdv2.storage import atomic_json, digest, read_json, run_lock
from . import policy
from .api import Gateway, ProviderError
from .examples import load_bank, pick_examples
from .selection import load_assignment, verified_images
from .image_inputs import resize_inputs


def image_block(data, detail):
    return {
        'type': 'image_url',
        'image_url': {
            'url': 'data:image/png;base64,' + base64.b64encode(data).decode(),
            'detail': detail,
        },
    }


def example_content(row, detail):
    from PIL import Image
    from io import BytesIO

    # Participant IDs, telemetry, and source paths are never sent to the provider.
    content = [
        {
            'type': 'text',
            'text': json.dumps(
                {
                    'example_base_prompt': row['prompt'],
                    'human_requests': row['requests'],
                    'human_no_variation_reason': row.get('none_reason', ''),
                    'example_images_supplied': bool(row.get('images')),
                },
                ensure_ascii=False,
            ),
        }
    ]
    if row.get('provenance') == 'human_derived_edited':
        content = [
            {
                'type': 'text',
                'text': json.dumps(
                    {
                        'example_base_prompt': row['prompt'],
                        'axes': row['requests'],
                        'example_provenance': 'Axes pooled across human responses and edited; may include requests made after model suggestions. List length is not a target axis count.',
                        'example_images_supplied': False,
                    },
                    ensure_ascii=False,
                ),
            }
        ]
    if row.get('provenance') == 'human_verbatim_pooled':
        content = [
            {
                'type': 'text',
                'text': json.dumps(
                    {
                        'example_base_prompt': row['prompt'],
                        'axes': row['requests'],
                        'example_provenance': 'Verbatim human requests pooled across responses, with similar requests deduplicated. Some followed model suggestions. List length is not a target axis count.',
                        'example_images_supplied': False,
                    },
                    ensure_ascii=False,
                ),
            }
        ]
    fingerprints = []
    for p in row.get('images', []):
        data = Path(p).read_bytes()
        with Image.open(BytesIO(data)) as im:
            if im.format != 'PNG':
                raise ValueError('Example images must be PNG')
            im.load()
        fingerprints.append(
            dict(path=str(Path(p).resolve()), sha256=hashlib.sha256(data).hexdigest())
        )
        content.append(image_block(data, detail))
    return content, fingerprints


def build_request(row, images, examples, config):
    images = resize_inputs(images, config.get('image_max_side'))
    messages = [{'role': 'system', 'content': policy.SYSTEM}]
    example_fingerprints = []
    # Examples are explicitly human data, not fabricated assistant observations.
    for e in examples:
        content, fingerprints = example_content(e, config['detail'])
        messages.append({'role': 'user', 'content': content})
        example_fingerprints.append(dict(id=e['id'], images=fingerprints))
    target = [
        {
            'type': 'text',
            'text': json.dumps(
                {
                    'base_prompt': row['prompt'],
                    'image_count': len(images),
                    'task': f"Provide up to {config['max_axes']} distinct paired axis candidates. Do not pad.",
                },
                ensure_ascii=False,
            ),
        }
    ]
    for index, image in enumerate(images):
        target.append({'type': 'text', 'text': f'Target image {index+1}'})
        target.append(image_block(image['data'], config['detail']))
    messages.append({'role': 'user', 'content': target})
    payload = dict(
        model=config['model'],
        messages=messages,
        max_completion_tokens=config['max_tokens'],
        response_format={
            'type': 'json_schema',
            'json_schema': {
                'name': 'axis_candidates',
                'strict': True,
                'schema': policy.schema(config['max_axes']),
            },
        },
    )
    if config.get('reasoning_effort') is not None:
        payload['reasoning_effort'] = config['reasoning_effort']
    # Temperature and an API seed are never sent.
    audit = dict(
        payload_hash=digest(payload),
        example_fingerprints=example_fingerprints,
        images=[{k: v for k, v in i.items() if k != 'data'} for i in images],
    )
    return payload, audit


def validate_config(config):
    allowed = {
        'api_base',
        'model',
        'reasoning_effort',
        'timeout',
        'detail',
        'max_tokens',
        'max_axes',
        'example_count',
        'seed',
    }
    if set(config) - {'image_max_side'} != allowed:
        raise ValueError('Unknown or missing API configuration fields')
    if 'image_max_side' in config and (
        type(config['image_max_side']) is not int or config['image_max_side'] < 1
    ):
        raise ValueError('image_max_side must be a positive integer')
    parsed = urlsplit(config['api_base'])
    if (
        parsed.scheme != 'https'
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError('Use an explicit HTTPS API base without credentials or query parameters')
    if not isinstance(config['model'], str) or not config['model'].strip():
        raise ValueError('Choose an explicit gateway model ID')
    for key in ('timeout', 'max_tokens', 'max_axes'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if config['example_count'] != 'all' and (
        type(config['example_count']) is not int or config['example_count'] < 1
    ):
        raise ValueError('example_count must be a positive integer or all')
    if type(config['seed']) is not int or config['detail'] not in {'low', 'high', 'auto'}:
        raise ValueError('Invalid sampling seed or image detail')
    if config['reasoning_effort'] not in {None, 'none', 'low', 'medium', 'high'}:
        raise ValueError('Unsupported reasoning effort; verify provider support')


def source_snapshot():
    root = Path(__file__).resolve().parent
    paths = list(root.glob('*.py')) + [
        root.parent / 'hpdv2' / 'storage.py',
        root.parent / 'baselines' / 'assets.py',
    ]
    return {str(p.relative_to(root.parent)): p.read_text() for p in sorted(paths)}


def run(
    assignment_path,
    examples_path,
    out,
    config,
    *,
    limit=40,
    execute=False,
    max_calls=None,
    api_key=None,
    retry_errors=False,
    retry_uncertain=False,
    client_factory=Gateway,
):
    validate_config(config)
    if type(limit) is not int or limit < 1:
        raise ValueError('Set a positive pilot limit explicitly')
    if execute and (type(max_calls) is not int or max_calls < 0):
        raise ValueError('Execution requires a cumulative API call budget')
    assignment = load_assignment(assignment_path)
    bank = load_bank(examples_path)
    # The assignment order was frozen before rendering finished.
    rows = assignment['items'][:limit]
    sources = source_snapshot()
    versions = {}
    for name in ('openai', 'Pillow'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    identity = dict(
        schema_version=1,
        assignment_hash=digest(assignment),
        bank_hash=digest(bank),
        config=config,
        ids=[r['id'] for r in rows],
        source_hash=digest(sources),
        policy_version=policy.VERSION,
        python=platform.python_version(),
        packages=versions,
    )
    out = Path(out)
    if not execute:
        ready = sum(verified_images(assignment, row) is not None for row in rows)
        for row in rows:
            pick_examples(bank, row, config['example_count'], config['seed'])
        return dict(
            execute=False,
            selected=len(rows),
            ready=ready,
            pending_images=len(rows) - ready,
            maximum_initial_calls=ready,
            identity_hash=digest(identity),
        )
    with run_lock(out):
        if (out / 'manifest.json').exists():
            if read_json(out / 'manifest.json') != identity:
                raise ValueError('Run identity changed; use a new output directory')
        elif any(p.name != '.lock' for p in out.iterdir()):
            raise ValueError('Artifacts exist without a run manifest')
        atomic_json(out / 'manifest.json', identity)
        atomic_json(out / 'source-snapshot.json', sources)
        atomic_json(out / 'examples-snapshot.json', bank)
        reservations = sorted((out / 'attempts').glob('*.json'))
        attempts = [read_json(p) for p in reservations]
        if [a['number'] for a in attempts] != list(range(1, len(attempts) + 1)):
            raise ValueError('API attempt journal is not contiguous')
        by_prompt = {a['prompt_id']: a for a in attempts}
        calls = len(attempts)
        if calls > max_calls:
            raise ValueError('Budget is smaller than already reserved calls')
        summary = dict(
            selected=len(rows),
            completed=0,
            pending_images=0,
            blocked=0,
            invalid=0,
            no_variation=0,
            unassessable=0,
            budget_pending=0,
            api_calls_reserved=calls,
            execution='serial pilot',
        )
        client = None

        def status():
            summary['api_calls_reserved'] = calls
            atomic_json(out / 'status.json', summary)

        status()
        for row in rows:
            images = verified_images(assignment, row)
            if images is None:
                summary['pending_images'] += 1
                status()
                continue
            examples = pick_examples(bank, row, config['example_count'], config['seed'])
            payload, audit = build_request(row, images, examples, config)
            folder = out / 'items' / row['id']
            request = dict(row=row, **audit)
            if (folder / 'request.json').exists() and read_json(folder / 'request.json') != request:
                raise ValueError('Prompt, examples or images changed after request creation')
            atomic_json(folder / 'request.json', request)
            reply_path = folder / 'reply.json'
            if not reply_path.exists():
                # The global attempt journal is authoritative, also after an interruption before
                # the per-item files were written.
                previous = by_prompt.get(row['id'])
                if previous:
                    failed = (folder / f"error-{previous['number']:06d}.json").exists()
                    if (failed and not retry_errors) or (not failed and not retry_uncertain):
                        summary['blocked'] += 1
                        status()
                        continue
                if calls >= max_calls:
                    summary['budget_pending'] += 1
                    status()
                    continue
                if client is None:
                    if not api_key:
                        raise ValueError('Missing API key; no requests sent')
                    client = client_factory(config, api_key)
                calls += 1
                attempt = dict(
                    number=calls, prompt_id=row['id'], payload_hash=audit['payload_hash']
                )
                # Reserve and fsync before dispatch, so a crash cannot hide a spent call.
                atomic_json(out / 'attempts' / f'{calls:06d}.json', attempt)
                atomic_json(folder / 'attempt.json', attempt)
                status()
                try:
                    reply = asdict(client(payload))
                except ProviderError as error:
                    atomic_json(folder / f'error-{calls:06d}.json', {'error': str(error)})
                    summary['blocked'] += 1
                    status()
                    continue
                atomic_json(reply_path, reply)  # Save provider text before parsing.
            reply = read_json(reply_path)
            if reply.get('refusal') or reply['finish_reason'] != 'stop':
                summary['blocked'] += 1
                status()
                continue
            try:
                parsed = policy.parse(reply['text'], config['max_axes'])
            except (ValueError, TypeError):
                summary['invalid'] += 1
                atomic_json(
                    folder / 'validation.json',
                    {'valid': False, 'reason': 'schema validation failed'},
                )
                status()
                continue
            atomic_json(
                folder / 'result.json',
                dict(
                    row=row,
                    response=parsed,
                    payload_hash=audit['payload_hash'],
                    actual_model=reply['model'],
                    usage=reply['usage'],
                    semantic_review='pending',
                ),
            )
            summary['completed'] += 1
            if parsed['status'] != 'ok':
                summary[parsed['status']] += 1
            status()
        summary['complete'] = summary['completed'] == len(rows)
        status()
        return summary
