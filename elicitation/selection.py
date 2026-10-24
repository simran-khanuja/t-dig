"""Freeze balanced T2I model assignments before looking at image availability."""

from collections import Counter
import hashlib
from pathlib import Path
import random

from hpdv2.storage import atomic_json, digest, prompt_id, read_json, run_lock

T2I_MODELS = ('flux2-klein-9b', 'sd35-medium', 'sd35-large', 'qwen-image-2512')


def assign(rows, per_t2i_model=5000, seed=42):
    if type(per_t2i_model) is not int or per_t2i_model < 1 or type(seed) is not int:
        raise ValueError('Require positive per-T2I-model count and integer seed')
    rows = sorted(rows, key=lambda r: r['id'])
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate prompt IDs')
    if any(r['id'] != prompt_id(r['prompt']) for r in rows):
        raise ValueError('Prompt ID/text mismatch')
    # IDs hash the exact text; also reject whitespace and case variants.
    normalized = [' '.join(r['prompt'].casefold().split()) for r in rows]
    if len(set(normalized)) != len(rows):
        raise ValueError('Repeated normalized prompt text')
    n = len(T2I_MODELS) * per_t2i_model
    if len(rows) < n:
        raise ValueError('Not enough unique prompts')
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    result = []
    for position, index in enumerate(indices[:n]):
        row = rows[index]
        result.append({k: row[k] for k in ('id', 'prompt', 'cluster_id') if k in row})
        result[-1].update(renderer=T2I_MODELS[position % len(T2I_MODELS)], shard=index % 8)
    return result, [rows[i]['id'] for i in indices[n:]]


def prepare(render_root, out, per_t2i_model=5000, seed=42):
    root, out = Path(render_root).resolve(), Path(out)
    plan = read_json(root / 'plan.json')
    plan_hash = digest(plan)
    if read_json(root / 'plan-hash.json') != {'sha256': plan_hash}:
        raise ValueError('T2I model plan hash mismatch')
    rows = read_json(root / 'prompts.json')
    if digest(rows) != plan['rows_sha256']:
        raise ValueError('T2I model prompts changed')
    if plan['k'] != 4 or set(plan['configs']) != set(T2I_MODELS):
        raise ValueError('Expected four-image, four-T2I-model training run')
    selected, unused = assign(rows, per_t2i_model, seed)
    manifests = {}
    for model in T2I_MODELS:
        manifests[model] = {}
        for shard in range(8):
            m = read_json(root / model / f'shard-{shard:02d}' / 'manifest.json')
            if (
                m['plan_hash'] != plan_hash
                or m['shard'] != shard
                or m['workers'] != 8
                or m['rows_sha256'] != digest(sorted(rows, key=lambda r: r['id'])[shard::8])
                or m['config'] != plan['configs'][model]
            ):
                raise ValueError('Original shard identity mismatch')
            manifests[model][str(shard)] = m
    value = dict(
        schema_version=1,
        kind='balanced_axis_assignment',
        algorithm='sorted-id-python-random-shuffle-roundrobin-v1',
        seed=seed,
        per_renderer=per_t2i_model,
        render_root=str(root),
        render_plan_hash=plan_hash,
        source_rows_hash=digest(rows),
        curation_decision=plan.get('curation_decision'),
        k=4,
        manifests=manifests,
        items=selected,
        unused_ids=unused,
    )
    with run_lock(out):
        path = out / 'assignment.json'
        if path.exists() and read_json(path) != value:
            raise ValueError('Assignment already exists with different settings')
        atomic_json(path, value)
        atomic_json(out / 'assignment-hash.json', {'sha256': digest(value)})
        atomic_json(
            out / 'summary.json',
            dict(
                prompts=len(selected),
                images=4 * len(selected),
                by_t2i_model=dict(Counter(r['renderer'] for r in selected)),
                unused=len(unused),
                image_readiness='not used for selection',
                assignment_hash=digest(value),
            ),
        )
    return value


def load_assignment(path):
    path = Path(path)
    value = read_json(path)
    if read_json(path.with_name('assignment-hash.json')) != {'sha256': digest(value)}:
        raise ValueError('Assignment integrity check failed')
    return value


def verified_images(assignment, row):
    """Return the row's verified images from its assigned T2I model, or None while any is pending.

    Each item holds `path`, `sha256`, `receipt` and the PNG bytes in `data`.
    """
    from baselines.assets import seed_for
    from PIL import Image
    from io import BytesIO

    root = Path(assignment['render_root']) / row['renderer'] / f"shard-{row['shard']:02d}"
    manifest = assignment['manifests'][row['renderer']][str(row['shard'])]
    if read_json(root / 'manifest.json') != manifest:
        raise ValueError('T2I model manifest changed')
    cfg = manifest['config']
    images = []
    for index in range(assignment['k']):
        slot = root / 'images' / row['id'] / f'{index:04d}'
        if not slot.exists() and not slot.is_symlink():
            return None  # T2I model commits are atomic, so an absent slot is pending.
        if (
            slot.is_symlink()
            or not slot.is_dir()
            or {p.name for p in slot.iterdir()} != {'image.png', 'receipt.json'}
            or any(p.is_symlink() for p in slot.iterdir())
        ):
            raise ValueError('Malformed committed image slot')
        receipt = read_json(slot / 'receipt.json')
        expected = dict(
            run_hash=digest(manifest),
            prompt_id=row['id'],
            index=index,
            seed=seed_for(row['id'], cfg['seed'], index),
            render_prompt=row['prompt'],
        )
        if any(receipt.get(k) != v for k, v in expected.items()):
            raise ValueError('Image receipt identity mismatch')
        raw = (slot / 'image.png').read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != receipt['sha256'] or len(raw) != receipt['bytes']:
            raise ValueError('Image checksum mismatch')
        with Image.open(BytesIO(raw)) as image:
            if image.format != 'PNG' or image.size != (cfg['resolution'], cfg['resolution']):
                raise ValueError('Unexpected image format or size')
            image.load()
        images.append(dict(path=str(slot / 'image.png'), sha256=sha, receipt=receipt, data=raw))
    return images
