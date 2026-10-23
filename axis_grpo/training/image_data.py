"""Build the image-bearing twin of a train/validation split for the image-input policy.

Each row of the twin is the text row plus the four i.i.d. images the elicitor saw for its base
prompt, resized to 512 px. `tdig/build_dataset.py` reads the twin (`--image-source`) for its
`flux2-klein-9b-images` variant and rewrites the `prompt` column, so the twin keeps the text
row's prompt. Rows, order, `reward_model` and every existing `extra_info` field are unchanged.

Inputs: the split (`axis_grpo.training.split`, `train` and `val` Parquet files and
`manifest.json`) and the `source-records.jsonl` of the export it was split from
(`axis_grpo.prepare build`), which names each prompt's four images with their receipts.

Outputs:
- `--image-cache/<prompt_id>/<i>.png`: RGB LANCZOS 512 x 512 PNG copies with a `manifest.json`
  per prompt. A rerun verifies and reuses them.
- `--out`: the two Parquet files with an `images` column (the four cache paths) and
  `extra_info.policy_observation = "elicitation_images"` and `extra_info.input_image_sha256`,
  plus `images.jsonl` (one cache manifest per prompt) and `manifest.json`.

Every source image is checked against its receipt and SHA-256 before it is copied.
"""

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path

from axis_grpo.dataset import export_parquet
from axis_grpo.prepare import new_directory
from baselines.assets import file_hash
from hpdv2.storage import atomic_json, read_json, run_lock

SPLITS = ('train', 'val')
SIZE = 512


def image_record(record, cache):
    """Write or verify the 512 px copies of one prompt's four images.

    Args:
        record: one line of `source-records.jsonl`, with `row` (`id`, `prompt`, `renderer`) and
            `images` (`path`, `sha256`, `receipt`).
        cache: image cache directory; the copies go to `<cache>/<prompt_id>/`.

    Returns the cache manifest, `dict(identity=..., images=[...])`, also written to
    `<cache>/<prompt_id>/manifest.json`. Raises `ValueError` when a receipt, checksum or image
    size disagrees, or when an existing cache entry was made from other sources or changed.
    """
    from PIL import Image

    row = record['row']
    out = Path(cache) / row['id']
    if len(record['images']) != 4:
        raise ValueError('Require exactly four elicitation images')
    identity = dict(
        prompt_id=row['id'],
        renderer=row['renderer'],
        prompt=row['prompt'],
        source_images=record['images'],
        transform=f'RGB LANCZOS {SIZE}x{SIZE} PNG',
    )
    manifest = out / 'manifest.json'
    if manifest.exists():
        saved = read_json(manifest)
        if saved['identity'] != identity:
            raise ValueError(f'Image cache identity changed: {out}')
        for item in saved['images']:
            if file_hash(item['path']) != item['sha256']:
                raise ValueError(f'Image cache bytes changed: {item["path"]}')
        return saved
    out.mkdir(parents=True, exist_ok=True)
    images = []
    for i, source in enumerate(record['images']):
        receipt = source['receipt']
        if (receipt['prompt_id'], receipt['index'], receipt['render_prompt']) != (
            row['id'],
            i,
            row['prompt'],
        ):
            raise ValueError('Image receipt does not match this prompt and slot')
        raw = Path(source['path']).read_bytes()
        if not hashlib.sha256(raw).hexdigest() == source['sha256'] == receipt['sha256']:
            raise ValueError(f'Source image checksum changed: {source["path"]}')
        with Image.open(BytesIO(raw)) as image:
            if image.format != 'PNG' or image.width != image.height:
                raise ValueError('Expected a square PNG')
            image = image.convert('RGB')
            image.thumbnail((SIZE, SIZE), Image.Resampling.LANCZOS)
            if image.size != (SIZE, SIZE):
                raise ValueError(f'Expected source images of at least {SIZE} px')
            path, temporary = out / f'{i}.png', out / f'.{i}.png'
            image.save(temporary, format='PNG')
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        images.append(
            dict(
                path=str(path.resolve()),
                sha256=file_hash(path),
                width=SIZE,
                height=SIZE,
                source_path=source['path'],
                source_sha256=source['sha256'],
            )
        )
    saved = dict(identity=identity, images=images)
    atomic_json(manifest, saved)
    return saved


def image_row(source, images):
    """Return a copy of one text row with its four images attached.

    Args:
        source: a row of the text split. Not modified.
        images: the cache manifest of the row's base prompt, from `image_record`.

    Raises `ValueError` when the prompt id, source T2I model or base prompt of the row and the
    images disagree.
    """
    row = copy.deepcopy(source)
    e = row['extra_info']
    identity = images['identity']
    if (identity['prompt_id'], identity['renderer'], identity['prompt']) != (
        e['prompt_id'],
        e['source_renderer'],
        e['base_prompt'],
    ):
        raise ValueError(f'Images do not belong to row {e["row_id"]}')
    row['images'] = [v['path'] for v in images['images']]
    e.update(
        policy_observation='elicitation_images',
        input_image_sha256=[v['sha256'] for v in images['images']],
    )
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--text-dataset', type=Path, required=True, help='train/validation split')
    p.add_argument('--records', type=Path, required=True, help='source-records.jsonl of the export')
    p.add_argument('--image-cache', type=Path, required=True, help='directory of 512 px copies')
    p.add_argument('--out', type=Path, required=True, help='new directory to publish')
    p.add_argument('--workers', type=int, default=16, help='threads resizing images')
    p.add_argument('--execute', action='store_true', help='write files; otherwise report counts')
    a = p.parse_args()
    import pyarrow.parquet as pq

    m = read_json(a.text_dataset / 'manifest.json')
    for name in (f'{s}.parquet' for s in SPLITS):
        if file_hash(a.text_dataset / name) != m['files'][name]:
            raise ValueError(f'Text dataset changed since its manifest: {name}')
    splits = {s: pq.read_table(a.text_dataset / f'{s}.parquet').to_pylist() for s in SPLITS}
    ids = {r['extra_info']['prompt_id'] for rows in splits.values() for r in rows}
    records = {}
    with a.records.open() as stream:
        for line in stream:
            r = json.loads(line)
            if r['row']['id'] in ids:
                records[r['row']['id']] = r
    if set(records) != ids:
        raise ValueError(f'{len(ids - set(records))} prompts have no record in {a.records}')
    if not a.execute:
        rows = {s: len(v) for s, v in splits.items()}
        print(json.dumps(dict(execute=False, prompts=len(ids), images=4 * len(ids), rows=rows)))
        return
    cache = a.image_cache.resolve()
    index = {}
    with run_lock(cache):
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            done = pool.map(lambda pid: image_record(records[pid], cache), sorted(ids))
            for i, saved in enumerate(done, 1):
                index[saved['identity']['prompt_id']] = saved
                if i % 1000 == 0 or i == len(ids):
                    print(f'{i}/{len(ids)} prompt image sets verified', flush=True)
        with new_directory(a.out, (a.text_dataset, cache)) as out:
            for name, rows in splits.items():
                data = [image_row(r, index[r['extra_info']['prompt_id']]) for r in rows]
                export_parquet(out / f'{name}.parquet', data)
                if pq.read_table(out / f'{name}.parquet').to_pylist() != data:
                    raise ValueError(f'{name}.parquet does not read back as written')
            with (out / 'images.jsonl').open('w') as stream:
                for pid in sorted(index):
                    stream.write(json.dumps(index[pid]) + '\n')
            summary = {k: v for k, v in m.items() if k not in ('files', 'source')}
            summary.update(
                source=str(a.text_dataset.resolve()),
                text_manifest_sha256=file_hash(a.text_dataset / 'manifest.json'),
                original_records_sha256=file_hash(a.records),
                policy_observation='elicitation_images',
                input_resolution=SIZE,
                input_images_per_prompt=4,
                unique_images=4 * len(ids),
            )
            summary['files'] = {f.name: file_hash(f) for f in out.iterdir() if f.is_file()}
            atomic_json(out / 'manifest.json', summary)


if __name__ == '__main__':
    main()
