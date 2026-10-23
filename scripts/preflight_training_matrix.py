"""CPU checks of frozen training settings, data, runtime and actual image processing."""

import argparse
import copy
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
from hpdv2.storage import atomic_json, read_json
from baselines.assets import file_hash
from axis_grpo.training.config import compose_config
from axis_grpo.training.order import validate_order_config
from axis_grpo.training.run import audit_data, runtime_identity


def lengths(cfg):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer, AutoProcessor
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.utils.tokenizer import build_multimodal_processor_inputs
    from omegaconf import OmegaConf

    tokenizer = AutoTokenizer.from_pretrained(cfg['policy_path'], local_files_only=True)
    visual = cfg['policy_observation'] == 'elicitation_images'
    rows = []
    for name in ('train', 'val'):
        rows.extend(pq.read_table(Path(cfg['train_file']).parent / f'{name}.parquet').to_pylist())
    extra = 0
    proof = {}
    if visual:
        processor = AutoProcessor.from_pretrained(cfg['policy_path'], local_files_only=True)
        dataset = object.__new__(RLHFDataset)
        dataset.image_key = 'images'
        dataset.video_key = 'videos'
        dataset.audio_key = 'audios'
        dataset.processor = processor
        row = copy.deepcopy(rows[0])
        messages = dataset._build_messages(row, 'prompt')
        raw = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        images, videos, audio = dataset._process_multi_modal_info(
            messages, processor.image_processor.patch_size, OmegaConf.create({})
        )
        inputs = build_multimodal_processor_inputs(
            processor, text=[raw], images=images, videos=videos, audio=audio
        )
        grids = inputs['image_grid_thw'].tolist()
        if len(grids) != 4 or any(image.size != (512, 512) for image in images):
            raise ValueError('Four 512px images must actually reach the processor')
        # Every cached image has the same size, so the image-token expansion is one constant,
        # measured here with the real processor and added to every row's length.
        extra = int(inputs['input_ids'].shape[-1]) - len(
            tokenizer(raw, add_special_tokens=False)['input_ids']
        )
        proof = dict(
            image_grids=grids,
            pixel_values_shape=list(inputs['pixel_values'].shape),
            processor_tokens=int(inputs['input_ids'].shape[-1]),
            image_expansion_tokens=extra,
        )
        texts = [
            processor.apply_chat_template(
                dataset._build_messages(copy.deepcopy(r), 'prompt'),
                add_generation_prompt=True,
                tokenize=False,
            )
            for r in rows
        ]
    else:
        texts = [
            tokenizer.apply_chat_template(r['prompt'], add_generation_prompt=True, tokenize=False)
            for r in rows
        ]
    maximum = 0
    for start in range(0, len(texts), 512):
        tokens = tokenizer(texts[start : start + 512], add_special_tokens=False)['input_ids']
        maximum = max(maximum, max(map(len, tokens)) + extra)
    if maximum > cfg['max_prompt_length']:
        raise ValueError(f'Policy input truncation: {maximum}>{cfg["max_prompt_length"]}')
    return dict(maximum=maximum, configured=cfg['max_prompt_length'], rows=len(rows), **proof)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--matrix', type=Path, required=True)
    a = p.parse_args()
    root = a.matrix.resolve()
    matrix = read_json(root / 'matrix.json')
    runtime = runtime_identity()
    if runtime != read_json(root / 'source/axis_grpo/training/runtime.lock.json'):
        raise ValueError('Runtime differs from frozen lock')
    for name, expected in read_json(root / 'source-hashes.json').items():
        if file_hash(root / 'source' / name) != expected:
            raise ValueError('Frozen source changed')
    checks = {}
    tokens = {}
    configs = {}
    audits = {}
    from verl.utils.config import validate_config

    for job in matrix['jobs']:
        run = Path(job['config']).parent
        cfg = read_json(run / 'train-config.json')
        for name in (
            'fleet-config.json',
            'train-config.json',
            'reward-config.json',
            'qualification-config.json',
            'renderer-assets.json',
        ):
            path = run / name
            configs[str(path.relative_to(root))] = file_hash(path)
        composed = compose_config(cfg, run / 'policy', False, 0)
        validate_order_config(composed)
        validate_config(composed, use_reference_policy=False, use_critic=False)
        key = cfg['train_file']
        if key not in audits:
            audits[key] = audit_data(cfg)
        checks[job['name']] = audits[key]
        # Chat templates can differ across policy sizes, so lengths are checked per job.
        tokens[job['name']] = lengths(cfg)
        print(job['name'], checks[job['name']], tokens[job['name']], flush=True)
    for name in ('matrix.json', 'qualification-train.parquet', 'qualification-val.parquet', 'validation.parquet'):
        if (root / name).exists():
            configs[name] = file_hash(root / name)
    atomic_json(
        root / 'preflight.json',
        dict(
            passed=True, runtime=runtime, checks=checks, token_lengths=tokens, config_hashes=configs
        ),
    )


if __name__ == '__main__':
    main()
