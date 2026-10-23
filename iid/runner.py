"""Strict manifests, independent seeded images, and per-image durable resumption."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import math
import os
from pathlib import Path
import platform
import uuid

from hpdv2.dataset import SCHEMA, validate_prompts
from hpdv2.storage import atomic_json, digest, read_json, run_lock, sync_directory
from .service import RenderService, SEED_MODULUS, INDEPENDENT_STRIDE


def load_pool(path, allow_partial=False):
    data = read_json(path)
    if data.get('schema_version') != SCHEMA or data.get('kind') != 'clustered_pool':
        raise ValueError(
            'Input must be a schema-2 clustered_pool, not old training Parquet or filter output'
        )
    if data.get('metadata', {}).get('upstream_complete') is not True and not allow_partial:
        raise ValueError('Incomplete curation: --allow-partial-input is for pilots only')
    rows = validate_prompts(data.get('kept'))
    for row in rows:
        if not isinstance(row.get('cluster_id'), str) or not row['cluster_id']:
            raise ValueError('Missing cluster ID')
    return data, sorted(rows, key=lambda r: r['id'])


def seeds_for(ident, seed, index):
    # Independent of input order, workers, k, and resume boundaries.
    base = int(digest({'prompt_id': ident, 'seed': seed}), 16) % SEED_MODULUS
    latent = (base + index * INDEPENDENT_STRIDE) % SEED_MODULUS
    return (latent - INDEPENDENT_STRIDE) % SEED_MODULUS, latent


def verify_png(path, resolution):
    from PIL import Image

    raw = Path(path).read_bytes()
    with Image.open(io.BytesIO(raw)) as im:
        if (
            im.format != 'PNG'
            or im.size != (resolution, resolution)
            or im.mode not in {'RGB', 'RGBA'}
        ):
            raise ValueError('Image must be RGB/RGBA PNG at the requested resolution')
        im.verify()
    with Image.open(io.BytesIO(raw)) as im:
        im.load()  # Full decode; verify() checks only PNG structure and CRCs.
    return hashlib.sha256(raw).hexdigest(), len(raw)


def identity(data, cfg):
    sources = {p.name: p.read_text() for p in sorted(Path(__file__).parent.glob('*.py'))}
    sources.update(
        {
            f'hpdv2/{name}.py': (Path(__file__).parents[1] / 'hpdv2' / f'{name}.py').read_text()
            for name in ['dataset', 'storage']
        }
    )
    return {
        'schema_version': 1,
        'kind': 'iid_manifest',
        'input_hash': digest(data),
        'config': cfg,
        'code_hash': digest(sources),
        'python': platform.python_version(),
        'pillow': importlib.metadata.version('Pillow'),
        'seed_contract': 'axis-framework-alpha1-single-image-v1',
    }


def inspect(run, rows, manifest):
    """Verify saved images against their receipts; return (valid records, pending (row, index)).

    Filenames alone are never trusted; a corrupt or untracked file raises instead of being
    overwritten.
    """
    cfg = manifest['config']
    valid = {}
    pending = []
    for row in rows:
        for i in range(cfg['k']):
            stem = run / 'images' / row['id'] / str(i)
            path = stem.with_suffix('.png')
            receipt = stem.with_suffix('.json')
            if not path.exists() and not receipt.exists():
                pending.append((row, i))
                continue
            if (
                not path.is_file()
                or not receipt.is_file()
                or path.is_symlink()
                or receipt.is_symlink()
            ):
                raise ValueError(
                    f'Missing or untracked image/receipt: {stem}; preserve it and inspect before retrying'
                )
            record = read_json(receipt)
            request_seed, latent_seed = seeds_for(row['id'], cfg['seed'], i)
            expected = {
                'run_hash': digest(manifest),
                'prompt_id': row['id'],
                'index': i,
                'request_seed': request_seed,
                'latent_seed': latent_seed,
                'path': str(path.relative_to(run)),
            }
            if any(record.get(k) != v for k, v in expected.items()):
                raise ValueError(f'Image receipt does not match run: {receipt}')
            sha, nbytes = verify_png(path, cfg['resolution'])
            if sha != record.get('sha256') or nbytes != record.get('bytes'):
                raise ValueError(f'Image content changed: {path}')
            valid[(row['id'], i)] = record
    return valid, pending


def write_summary(run, rows, manifest, valid, errors):
    k = manifest['config']['k']
    items = []
    pending = []
    for row in rows:
        images = [valid[(row['id'], i)] for i in range(k) if (row['id'], i) in valid]
        if len(images) == k:
            # Do not leak pass-two screening axes into later image-based elicitation.
            items.append(
                {
                    'id': row['id'],
                    'prompt': row['prompt'],
                    'cluster_id': row['cluster_id'],
                    **({'split': row['split']} if 'split' in row else {}),
                    'images': images,
                }
            )
        else:
            pending.append(
                {
                    'id': row['id'],
                    'missing_indices': [i for i in range(k) if (row['id'], i) not in valid],
                }
            )
    result = {
        'schema_version': 1,
        'kind': 'iid_summary',
        'run_hash': digest(manifest),
        'metadata': {
            'k': k,
            'n_prompts': len(rows),
            'n_complete': len(items),
            'n_images': len(valid),
            'n_pending_images': len(rows) * k - len(valid),
            'complete': not pending,
            'upstream_complete': manifest['upstream_complete'],
            'full_pipeline_complete': not pending and manifest['upstream_complete'],
        },
        'items': items,
        'pending': pending,
        'errors': errors,
    }
    atomic_json(run / 'summary.json', result)
    return result


def generate(
    input_path,
    run_dir,
    cfg,
    *,
    execute=False,
    max_requests=None,
    limit=None,
    allow_partial=False,
    client=None,
):
    for name in ['k', 'steps', 'resolution']:
        if type(cfg[name]) is not int or cfg[name] <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if cfg['k'] > SEED_MODULUS:
        raise ValueError('k exceeds the T2I model seed space')
    if not math.isfinite(cfg['guidance']) or cfg['guidance'] < 0:
        raise ValueError('Guidance must be finite and nonnegative')
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError('limit must be nonnegative')
    if execute and (type(max_requests) is not int or max_requests <= 0):
        raise ValueError('--execute requires a positive --max-requests')
    data, rows = load_pool(input_path, allow_partial)
    run = Path(run_dir).resolve()
    manifest = identity(data, cfg)
    manifest['upstream_complete'] = data.get('metadata', {}).get('upstream_complete') is True
    manifest['allow_partial_input'] = allow_partial

    def check():
        if (run / 'manifest.json').exists():
            if read_json(run / 'manifest.json') != manifest:
                raise ValueError(
                    'Manifest mismatch; use a new output directory for changed input/settings/code'
                )
        elif run.exists() and any(p.name != '.lock' for p in run.iterdir()):
            raise ValueError('Existing run artifacts lack a manifest')
        return inspect(run, rows, manifest)

    if not execute:
        valid, pending = check()
        wanted = {r['id'] for r in rows if any((r['id'], i) not in valid for i in range(cfg['k']))}
        n = len(wanted) if limit is None else min(limit, len(wanted))
        return {
            'execute': False,
            'k': cfg['k'],
            'n_prompts': len(rows),
            'n_images_total': len(rows) * cfg['k'],
            'n_verified_images': len(valid),
            'n_pending_images': len(pending),
            'n_selected_prompts': n,
        }
    with run_lock(run):
        valid, pending = check()
        if not (run / 'manifest.json').exists():
            atomic_json(run / 'manifest.json', manifest)
            atomic_json(run / 'input.json', data)
        selected = list(dict.fromkeys(r['id'] for r, i in pending))
        if limit is not None:
            selected = selected[:limit]
        selected = set(selected)
        pending = [(r, i) for r, i in pending if r['id'] in selected]
        errors = []
        result = write_summary(run, rows, manifest, valid, errors)
        if not pending:
            return result['metadata']
        client = client or RenderService(cfg['render_url'], cfg['timeout'])
        health = client.health()
        if (
            health.get('ready') is not True
            or health.get('fake') is not False
            or cfg['model'] not in health.get('models', [])
        ):
            raise ValueError(
                'T2I model must be ready, real (not fake), and serve the requested model'
            )
        used = 0
        for row, i in pending:
            if used >= max_requests:
                break
            request_seed, latent_seed = seeds_for(row['id'], cfg['seed'], i)
            stage = run / 'staging' / uuid.uuid4().hex
            stage.mkdir(parents=True)
            payload = {
                'prompts': [row['prompt']],
                'base_prompt': row['prompt'],
                'out_dir': str(stage),
                'tag': 'image',
                'model': cfg['model'],
                'seed': request_seed,
                'init_diversity': 1.0,
                'steps': cfg['steps'],
                'resolution': cfg['resolution'],
                'guidance_scale': cfg['guidance'],
                'index_offset': 0,
                'n_total': 1,
            }
            used += 1
            try:
                response = client.render(payload)
                atomic_json(stage / 'response.json', response)
                fields = [
                    'prompts',
                    'seed',
                    'model',
                    'init_diversity',
                    'steps',
                    'resolution',
                    'guidance_scale',
                    'index_offset',
                    'n_total',
                ]
                if response.get('fake') is not False or any(
                    response.get(f) != payload[f] for f in fields
                ):
                    raise ValueError('T2I model response does not match requested settings')
                if response.get('model_path') != cfg['model_path']:
                    raise ValueError('T2I model checkpoint path mismatch')
                expected = stage / 'image_0.png'
                if response.get('images') != [str(expected)] or expected.is_symlink():
                    raise ValueError('T2I model must return exactly the requested local image path')
                sha, nbytes = verify_png(expected, cfg['resolution'])
                dest = run / 'images' / row['id'] / f'{i}.png'
                dest.parent.mkdir(parents=True, exist_ok=True)
                with expected.open('rb') as stream:
                    os.fsync(stream.fileno())
                os.replace(expected, dest)
                sync_directory(dest.parent)
                record = {
                    'run_hash': digest(manifest),
                    'prompt_id': row['id'],
                    'index': i,
                    'path': str(dest.relative_to(run)),
                    'sha256': sha,
                    'bytes': nbytes,
                    'request_seed': request_seed,
                    'latent_seed': latent_seed,
                    'response_path': str((stage / 'response.json').relative_to(run)),
                    'created': datetime.now(timezone.utc).isoformat(),
                }
                atomic_json(dest.with_suffix('.json'), record)
                valid[(row['id'], i)] = record
            except Exception as error:
                errors.append(
                    {
                        'prompt_id': row['id'],
                        'index': i,
                        'error': type(error).__name__,
                        'staging_path': str(stage.relative_to(run)),
                    }
                )
                write_summary(run, rows, manifest, valid, errors)
                raise ValueError(
                    f'Render failed for {row["id"]}/{i} ({type(error).__name__}); inspect staging; completed images preserved'
                ) from None
            if used % 100 == 0:
                write_summary(run, rows, manifest, valid, errors)
            print(f'Verified {len(valid)}/{len(rows)*cfg["k"]} images', flush=True)
        result = write_summary(run, rows, manifest, valid, errors)
        return {**result['metadata'], 'requests_this_invocation': used}
