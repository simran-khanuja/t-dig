"""Create a shared, prompt-disjoint train/validation export without changing sources."""

import argparse
import copy
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from baselines.assets import file_hash
from hpdv2.storage import atomic_json, read_json
from axis_grpo.prepare import new_directory
from axis_grpo.dataset import export_parquet


def split_rows(rows, per_t2i_model=50, seed=42):
    if type(per_t2i_model) is not int or per_t2i_model < 1:
        raise ValueError('Require a positive per-T2I-model holdout size')
    groups = defaultdict(dict)
    for row in rows:
        e = row['extra_info']
        groups[e['source_renderer']][e['prompt_id']] = e
    if len(groups) != 4:
        raise ValueError('Require all four source T2I models')
    rng = random.Random(seed)
    chosen = set()
    for t2i_model in sorted(groups):
        ids = sorted(groups[t2i_model])
        if len(ids) <= per_t2i_model:
            raise ValueError('Insufficient prompts for disjoint train/validation')
        chosen.update(rng.sample(ids, per_t2i_model))
    train, val = [], []
    for source in rows:
        row = copy.deepcopy(source)
        e = row['extra_info']
        target = val if e['prompt_id'] in chosen else train
        e['source_export_index'] = e['index']
        e['index'] = len(target)
        e['split'] = 'val' if target is val else 'train'
        target.append(row)
    for key in ('prompt_id', 'base_prompt', 'cluster_id'):
        left = {r['extra_info'][key] for r in train} - {None}
        right = {r['extra_info'][key] for r in val} - {None}
        if left & right:
            raise ValueError(f'Leaking {key} across train/validation')
    return train, val, sorted(chosen)


def val_subset(rows, axes_per_prompt):
    """Return the `axes_per_prompt` lowest-slot named axes of each prompt plus its overall rows."""
    if type(axes_per_prompt) is not int or axes_per_prompt < 1:
        raise ValueError('Require a positive number of axes per prompt')
    slots = defaultdict(list)
    for r in rows:
        e = r['extra_info']
        if e['axis_kind'] == 'axis':
            slots[e['prompt_id']].append(e['axis_slot'])
    kept = {pid: sorted(s)[:axes_per_prompt] for pid, s in slots.items()}
    subset = [
        copy.deepcopy(r)
        for r in rows
        if r['extra_info']['axis_kind'] == 'overall'
        or r['extra_info']['axis_slot'] in kept[r['extra_info']['prompt_id']]
    ]
    for i, r in enumerate(subset):
        r['extra_info']['index'] = i
    return subset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--per-t2i-model', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    import pyarrow.parquet as pq

    manifest = read_json(a.source / 'manifest.json')
    if not manifest['complete']:
        raise ValueError('Source dataset is incomplete')
    for name, expected in manifest['files'].items():
        if file_hash(a.source / name) != expected:
            raise ValueError(f'Source changed: {name}')
    rows = pq.read_table(a.source / 'train.parquet').to_pylist()
    train, val, ids = split_rows(rows, a.per_t2i_model, a.seed)
    report = dict(
        complete=True,
        source=str(a.source.resolve()),
        source_manifest_sha256=file_hash(a.source / 'manifest.json'),
        seed=a.seed,
        rows=len(train),
        val_rows=len(val),
        base_prompts=len({r['extra_info']['prompt_id'] for r in train}),
        val_base_prompts=len(ids),
        overall_rows=sum(r['extra_info']['axis_kind'] == 'overall' for r in train),
        val_source_renderers=dict(
            Counter(
                next(
                    r['extra_info']['source_renderer']
                    for r in val
                    if r['extra_info']['prompt_id'] == pid
                )
                for pid in ids
            )
        ),
        reward_text='request',
        split_unit='base prompt with all axes and overall rows',
    )
    print(json.dumps(report, indent=2))
    if not a.execute:
        return
    with new_directory(a.out, (a.source,)) as out:
        for name, data in [('train', train), ('val', val)]:
            export_parquet(out / (name + '.parquet'), data)
            # Read the file back to confirm text and metadata survive the Arrow roundtrip.
            if pq.read_table(out / (name + '.parquet')).to_pylist() != data:
                raise RuntimeError('Parquet export/readback differs')
        atomic_json(out / 'validation-prompt-ids.json', ids)
        (out / 'README.md').write_text(
            '# Shared policy train/validation split\n\n'
            f"{report['base_prompts']:,} training prompts ({len(train):,} rows); "
            f"{len(ids)} held-out prompts ({len(val)} rows).\n\n"
            'val.parquet holds every held-out axis. Source text and axis wording are unchanged. '
            'Prompt IDs, text and source clusters are disjoint; training pass order is preserved.\n'
        )
        report['files'] = {f.name: file_hash(f) for f in out.iterdir() if f.is_file()}
        atomic_json(out / 'manifest.json', report)


if __name__ == '__main__':
    main()
