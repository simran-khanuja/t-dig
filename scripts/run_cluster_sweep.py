"""Frozen remote HPDv2 clustering sweep, with resumable embeddings and linkage."""

import argparse
import os
from pathlib import Path
import shutil
import socket
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hpdv2.dataset import load_input
from hpdv2.storage import atomic_json, digest, read_json, run_lock, scheduler_job_id
from baselines.assets import inventory, file_hash

ROOT = Path(__file__).resolve().parents[1]


def prepare(out, input_path, model_path, target_path):
    if out.exists():
        raise ValueError('Use a new sweep directory')
    data, _ = load_input(input_path, 'cluster', True)
    if not data['metadata']['complete']:
        raise ValueError('Wait until pass 2 has decided every input')
    if not data['kept']:
        raise ValueError('No accepted prompts')
    out.mkdir(parents=True)
    atomic_json(out / 'input.json', data)
    atomic_json(
        out / 'unresolved.json',
        dict(
            pass2_pending=data['pending'],
            pass2_errors=data['errors'],
            treatment='Excluded from this provisional sweep; original decisions are not modified',
        ),
    )
    for package in ['hpdv2', 'baselines']:
        (out / 'source' / package).mkdir(parents=True)
        for p in (ROOT / package).glob('*.py'):
            shutil.copy2(p, out / 'source' / package / p.name)
    (out / 'source/scripts').mkdir()
    shutil.copy2(__file__, out / 'source/scripts/run_cluster_sweep.py')
    shutil.copy2(ROOT / 'scripts/launch_cluster_sweep.sh', out / 'launch.sh')
    provenance = read_json(model_path / 'tdig-provenance.json')
    plan = dict(
        input_hash=digest(data),
        input_path=str(input_path),
        n_prompts=len(data['kept']),
        thresholds=[0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
        seed=42,
        model=provenance['repository'],
        revision=provenance['revision'],
        model_inventory=inventory(model_path),
        dependency_inventory=inventory(target_path),
        source_hashes={
            str(p.relative_to(out)): file_hash(p) for p in (out / 'source').rglob('*.py')
        },
        launcher_sha256=file_hash(out / 'launch.sh'),
        max_matrix_gib=64,
        purpose='Threshold calibration; not a selected final training pool',
    )
    atomic_json(out / 'plan.json', plan)
    atomic_json(out / 'plan-hash.json', dict(sha256=digest(plan)))
    atomic_json(
        out / 'status.json',
        dict(state='prepared', n_prompts=plan['n_prompts'], thresholds=plan['thresholds']),
    )
    return dict(n_prompts=plan['n_prompts'], matrix_gib=plan['n_prompts'] ** 2 * 8 / 2**30)


def save_array(out, name, writer):
    target = out / name
    tmp = out / ('.' + name + '.tmp')
    with tmp.open('wb') as f:
        writer(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target)
    atomic_json(out / (name + '.receipt.json'), dict(sha256=file_hash(target)))


def checked_array(out, name):
    target = out / name
    if not target.exists():
        return None
    if (
        not (out / (name + '.receipt.json')).exists()
        or file_hash(target) != read_json(out / (name + '.receipt.json'))['sha256']
    ):
        raise ValueError('Unverified/corrupt cached array: ' + name)
    import numpy as np

    return np.load(target, allow_pickle=False)


def run(out):
    plan = read_json(out / 'plan.json')
    if digest(plan) != read_json(out / 'plan-hash.json')['sha256']:
        raise ValueError('Plan changed')
    if Path(__file__).resolve() != out / 'source/scripts/run_cluster_sweep.py':
        raise ValueError('Use frozen code')
    for p, sha in plan['source_hashes'].items():
        if file_hash(out / p) != sha:
            raise ValueError('Source changed')
    if file_hash(out / 'launch.sh') != plan['launcher_sha256']:
        raise ValueError('Launcher changed')
    for key in ['model_inventory', 'dependency_inventory']:
        if inventory(plan[key]['path']) != plan[key]:
            raise ValueError(key + ' changed')
    data, _ = load_input(out / 'input.json', 'cluster', True)
    if digest(data) != plan['input_hash']:
        raise ValueError('Input changed')
    records = sorted(data['kept'], key=lambda r: r['id'])
    n = len(records)
    if n**2 * 8 / 2**30 > plan['max_matrix_gib']:
        raise ValueError('Distance matrix budget exceeded')
    import numpy as np
    from hpdv2.cluster_sweep import full_tree, cut_tree, pool_at, write_review

    with run_lock(out):

        def status(state, **extra):
            atomic_json(
                out / 'status.json',
                dict(
                    state=state,
                    pid=os.getpid(),
                    host=socket.gethostname(),
                    job_id=scheduler_job_id(),
                    updated=time.time(),
                    n_prompts=n,
                    **extra,
                ),
            )

        try:
            status('loading_embeddings')
            vectors = checked_array(out, 'embeddings.npy')
            if vectors is None:
                from sentence_transformers import SentenceTransformer
                import torch

                torch.manual_seed(plan['seed'])
                model = SentenceTransformer(
                    plan['model_inventory']['path'], device='cuda:0', local_files_only=True
                )
                status('embedding')
                vectors = np.asarray(
                    model.encode(
                        [r['prompt'] for r in records],
                        batch_size=256,
                        normalize_embeddings=True,
                        show_progress_bar=True,
                    ),
                    dtype=np.float64,
                )
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                if not np.isfinite(vectors).all() or (norms == 0).any():
                    raise ValueError('Invalid embeddings')
                vectors /= norms
                save_array(out, 'embeddings.npy', lambda f: np.save(f, vectors, allow_pickle=False))
                del model
                torch.cuda.empty_cache()
            if vectors.ndim != 2 or len(vectors) != n or not np.isfinite(vectors).all():
                raise ValueError('Embedding shape/content')
            status('building_hierarchy', matrix_gib=n * n * 8 / 2**30)
            tree = checked_array(out, 'hierarchy.npz')
            if tree is None:
                children, heights = full_tree(vectors, progress=status)
                save_array(
                    out, 'hierarchy.npz', lambda f: np.savez(f, children=children, heights=heights)
                )
            else:
                children, heights = tree['children'], tree['heights']
            import importlib.metadata, platform

            provenance = dict(
                model=plan['model'],
                revision=plan['revision'],
                plan_hash=digest(plan),
                embeddings_sha256=file_hash(out / 'embeddings.npy'),
                hierarchy_sha256=file_hash(out / 'hierarchy.npz'),
                python=platform.python_version(),
                packages={
                    p: importlib.metadata.version(p)
                    for p in [
                        'sentence-transformers',
                        'transformers',
                        'torch',
                        'numpy',
                        'scipy',
                        'scikit-learn',
                    ]
                },
            )
            stats, pairs = [], []
            for threshold in plan['thresholds']:
                status('cutting_threshold', distance=threshold, completed_thresholds=len(stats))
                labels = cut_tree(children, heights, n, threshold)
                pool, stat, audit = pool_at(
                    data, records, vectors, labels, threshold, provenance, plan['seed']
                )
                atomic_json(out / f'clustered-distance-{threshold:.2f}.json', pool)
                stats.append(stat)
                pairs.extend(audit)
                atomic_json(
                    out / 'sweep.json',
                    dict(thresholds=stats, provenance=provenance, final_threshold=None),
                )
                del pool
            # A seeded shuffle keeps the threshold order out of the annotation order.
            import random

            random.Random(plan['seed']).shuffle(pairs)
            atomic_json(out / 'audit-pairs.json', pairs)
            write_review(out, stats, pairs)
            status(
                'complete_pending_threshold_review',
                thresholds=stats,
                audit_pairs=len(pairs),
                final_threshold=None,
                upstream_complete=data['metadata']['full_pipeline_complete'],
            )
        except Exception as e:
            status('failed', error=f'{type(e).__name__}: {e}')
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'run'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--input', type=Path)
    p.add_argument(
        '--model',
        type=Path,
        default=Path(os.environ.get('T_DIG_MODELS', ROOT / 'models')) / 'all-MiniLM-L6-v2',
    )
    p.add_argument('--dependencies', type=Path, default=ROOT / '.cache/cluster-python')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if not a.execute:
        p.error('Explicit --execute required; preparation runs no GPU work')
    if a.action == 'prepare':
        print(
            prepare(a.out.resolve(), a.input.resolve(), a.model.resolve(), a.dependencies.resolve())
        )
    else:
        run(a.out.resolve())


if __name__ == '__main__':
    main()
