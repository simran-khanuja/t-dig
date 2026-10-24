"""Bounded T2I model/scorer integration probe; never optimizes a policy."""

import argparse
import json
from pathlib import Path
import time
import numpy as np

from hpdv2.storage import atomic_json, read_json
from .protocol import QUALITY_SCORER, seed_for


def fastest(measurements):
    usable = [v for v in measurements if 'images_per_second' in v]
    if not usable:
        raise ValueError('The T2I model does not fit at the smallest render batch')
    return max(usable, key=lambda v: v['images_per_second'])


def probe(config, out):
    import torch
    from .backend import GPUBackend
    from .metrics import score

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    backend = GPUBackend(config)
    base = 'A red chair beside a window'
    item = dict(
        base_prompt=base,
        axis='Chair material',
        tpips_factor='chair material',
        k=4,
        prompts=[
            f'A red {m} chair beside a window' for m in ('wooden', 'metal', 'plastic', 'velvet')
        ],
        seed=seed_for(base, 'chair material'),
        split='train',
        row_id='renderer-probe',
    )
    measurements = []
    reference = None
    # Each batch size renders the same 16 images with the same seeds: one warmup, one timed run.
    for batch in (4, 8, 16):
        backend.render_batch = batch
        try:
            backend.render([item] * 4)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            groups = backend.render([item] * 4)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            measurement = dict(
                batch_size=batch,
                images=16,
                seconds=elapsed,
                images_per_second=16 / elapsed,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
            pixels = np.stack(
                [np.asarray(image, dtype=np.float32) for group in groups for image in group]
            )
            if reference is None:
                reference = pixels
            measurement['mean_absolute_pixel_difference_vs_batch4'] = float(
                np.abs(pixels - reference).mean()
            )
            measurements.append(measurement)
        except torch.cuda.OutOfMemoryError:
            measurements.append(dict(batch_size=batch, error='out_of_memory'))
            torch.cuda.empty_cache()
            break
        atomic_json(out / 'render-benchmark.json', measurements)
    best = fastest(measurements)
    backend.render_batch = best['batch_size']
    groups = backend.render([item])
    images = groups[0]
    for i, image in enumerate(images):
        image.save(out / f'image-{i}.png')
    embeddings = []
    for batch in [1, 4]:
        backend.tpips_batch = batch
        start = time.monotonic()
        vectors = backend.embed(images, item['tpips_factor'])
        torch.cuda.synchronize()
        embeddings.append(dict(batch=batch, seconds=time.monotonic() - start, vectors=vectors))
    delta = float(np.max(np.abs(embeddings[0]['vectors'] - embeddings[1]['vectors'])))
    if delta > 0.01:
        raise RuntimeError(f'TPIPS batch parity exceeded tolerance: {delta}')
    alignment = backend.alignment(groups, [item])[0]
    # The mean of the item's own HPSv3 scores stands in for quality_thresh, so the quality term
    # runs and costs nothing.
    quality = {}
    if 'quality_weight' in config['reward']:
        values = backend.tracked(images, item['base_prompt'])[QUALITY_SCORER]
        quality = dict(quality=values, quality_threshold=sum(values) / len(values))
    rewards = score(
        embeddings[-1]['vectors'],
        alignment,
        item['tpips_factor'],
        config['reward'],
        prompts=item['prompts'],
        **quality,
    )
    # Identical prompts with the set's shared seed must give the same image, within 1 pixel level.
    duplicate = dict(item, prompts=[item['prompts'][0]] * 4)
    repeated = backend.render([duplicate])[0]
    max_delta = max(
        int(np.abs(np.asarray(im, dtype=np.int16) - np.asarray(repeated[0], dtype=np.int16)).max())
        for im in repeated
    )
    if max_delta > 1:
        raise RuntimeError(f'Shared-noise identity check failed: {max_delta}')
    validation = []
    for batch in (1, 4):
        backend.config['val_render_batch'] = batch
        val_item = dict(item, split='val', row_id='validation-renderer-probe')
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.monotonic()
            val_images = backend.render([val_item])[0]
            torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            validation.append(
                dict(
                    batch_size=batch,
                    images=4,
                    seconds=elapsed,
                    images_per_second=4 / elapsed,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                )
            )
            # Also run both scorers at the validation resolution.
            backend.alignment([val_images], [val_item])
            backend.embed(val_images, val_item['tpips_factor'])
        except torch.cuda.OutOfMemoryError:
            validation.append(dict(batch_size=batch, error='out_of_memory'))
            torch.cuda.empty_cache()
            break
    val_best = fastest(validation)
    result = dict(
        renderer=config['renderer'],
        settings=config['render'],
        selected_render_batch=best['batch_size'],
        selected_val_render_batch=val_best['batch_size'],
        validation_benchmark=validation,
        benchmark=measurements,
        shared_noise_max_pixel_difference=max_delta,
        tpips_batch_max_abs_difference=delta,
        tpips_timings=[{k: v for k, v in e.items() if k != 'vectors'} for e in embeddings],
        alignment=alignment,
        reward=rewards,
        device=torch.cuda.get_device_name(),
        status='passed',
    )
    atomic_json(out / 'summary.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if not a.execute:
        print(json.dumps(dict(execute=False)))
        return
    print(json.dumps(probe(read_json(a.config), a.out), indent=2))


if __name__ == '__main__':
    main()
