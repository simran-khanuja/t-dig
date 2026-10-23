"""Validate or run one generation policy. Does not start image services or call APIs."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import urllib.request

from baselines.assets import file_hash
from hpdv2.storage import atomic_json, digest, read_json, run_lock
from .config import settings, compose_config
from .order import PhaseBatchSampler, validate_order_config
from .protocol import parse_spec


def audit_data(cfg):
    import pyarrow.parquet as pq

    train = pq.read_table(cfg['train_file']).to_pylist()
    val = pq.read_table(cfg['val_file']).to_pylist()
    if not train or not val:
        raise ValueError('Require nonempty training and validation datasets')
    sampler = PhaseBatchSampler([r['extra_info'] for r in train], cfg['batch_size'])
    checked_images = {}
    for split, rows in [('train', train), ('val', val)]:
        for r in rows:
            spec = parse_spec(r['reward_model']['ground_truth'])
            if r['data_source'] != 'axis_diversity' or r['extra_info']['split'] != split:
                raise ValueError('Wrong data source or split marker')
            if (
                spec['axis'] != r['extra_info']['axis']
                or spec['base_prompt'] != r['extra_info']['base_prompt']
            ):
                raise ValueError('Reward ground truth differs from row metadata')
            images = r.get('images', [])
            if cfg['policy_observation'] == 'elicitation_images':
                if len(images) != 4 or sum(m['content'].count('<image>') for m in r['prompt']) != 4:
                    raise ValueError('Image count/placeholder mismatch')
                hashes = r['extra_info']['input_image_sha256']
                if len(hashes) != 4:
                    raise ValueError('Missing image hashes')
                for path, expected in zip(images, hashes):
                    if path not in checked_images:
                        checked_images[path] = file_hash(path)
                    if checked_images[path] != expected:
                        raise ValueError('Policy input image changed')
            elif images:
                raise ValueError('Image-bearing rows require an explicit image-input configuration')
    for key in ['base_prompt', 'prompt_id', 'cluster_id']:
        a = {r['extra_info'].get(key) for r in train} - {None}
        b = {r['extra_info'].get(key) for r in val} - {None}
        if a & b:
            raise ValueError(f'Training/validation overlap: {key}')
    if not cfg['smoke']:
        manifest = read_json(Path(cfg['train_file']).parent / 'manifest.json')
        # Outside smoke tests, train_file must be the complete export listed, with its hash,
        # in the dataset builder's manifest.json.
        if (
            manifest.get('complete') is not True
            or manifest.get('rows') != len(train)
            or manifest.get('files', {}).get(Path(cfg['train_file']).name)
            != file_hash(cfg['train_file'])
        ):
            raise ValueError('Training file is not the hashed dataset-builder export')
    return dict(
        train_rows=len(train), val_rows=len(val), steps=len(sampler), curriculum=sampler.identity
    )


def runtime_identity():
    import verl

    root = Path(verl.__file__).parent
    files = [
        'trainer/main_ppo.py',
        'trainer/ppo/v1/trainer_base.py',
        'trainer/ppo/v1/trainer_sync.py',
        'trainer/ppo/v1/replay_buffer.py',
        'experimental/reward_loop/reward_manager/naive.py',
    ]
    return dict(
        versions={
            name: importlib.metadata.version(name)
            for name in ['verl', 'torch', 'vllm', 'transformers', 'ray', 'torchdata']
        },
        framework_files={name: file_hash(root / name) for name in files},
    )


def stable_identity(identity):
    """Return the identity compared on resume.

    The step cap is dropped. With allow_endpoint_rebind, endpoint URLs become slot names.
    """
    import copy

    result = copy.deepcopy(identity)
    # The step cap may be raised to extend a finished run; the curriculum hash still pins the rows.
    result['config'].pop('max_steps', None)
    if result['config'].get('allow_endpoint_rebind'):
        urls = result['config']['reward_endpoints']
        result['config']['reward_endpoints'] = [f'reward-slot-{i}' for i in range(len(urls))]
        if 'reward_services' in result:
            result['reward_services'] = {
                f'reward-slot-{i}': result['reward_services'][url.rstrip('/')]
                for i, url in enumerate(urls)
            }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument(
        '--resume-from', type=Path, help='An earlier complete checkpoint within this same run'
    )
    p.add_argument('--stop-after-step', type=int, default=0)
    a = p.parse_args()
    cfg = settings(read_json(a.config))
    # An interpreter started by absolute path does not put its environment's bin directory on
    # PATH; vLLM and FlashInfer need the ninja executable there for CUDA compilation.
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    os.environ.update(
        HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',
        HF_DATASETS_OFFLINE='1',
        HF_HUB_CACHE=cfg['cache'],
        HF_DATASETS_CACHE=str(a.out.resolve() / 'cache/datasets'),
    )
    if a.stop_after_step and not cfg['smoke']:
        raise ValueError('Step cap is only for explicit smoke tests')
    if a.stop_after_step < 0:
        raise ValueError('Negative smoke-test step cap')
    audit = audit_data(cfg)
    runtime = runtime_identity()
    locked = read_json(Path(__file__).with_name('runtime.lock.json'))
    if runtime != locked:
        raise ValueError(
            'Trainer runtime differs from the inspected lock; re-audit the V1 adapter before use'
        )
    source = Path(__file__).parent
    identity = dict(
        config=cfg,
        data=audit,
        runtime=runtime,
        train_sha256=file_hash(cfg['train_file']),
        val_sha256=file_hash(cfg['val_file']),
        source={x.name: file_hash(x) for x in sorted(source.glob('*.py'))},
    )
    composed = compose_config(cfg, a.out, a.resume, a.stop_after_step)
    if a.resume_from:
        if not a.resume or a.resume_from.resolve().parent != (a.out / 'checkpoints').resolve():
            raise ValueError('--resume-from requires --resume and a checkpoint inside this run')
        composed.trainer.resume_mode = 'resume_path'
        composed.trainer.resume_from_path = str(a.resume_from.resolve())
    validate_order_config(composed)
    from verl.utils.config import validate_config

    validate_config(config=composed, use_reference_policy=False, use_critic=False)
    if not a.execute:
        print(json.dumps(dict(execute=False, **identity), indent=2))
        return
    from omegaconf import OmegaConf
    from verl.trainer.main_ppo import run_ppo
    from .trainer import TaskRunner

    os.environ['WANDB_MODE'] = 'offline'
    services = {}
    for url in cfg['reward_endpoints']:
        with urllib.request.urlopen(url.rstrip('/') + '/health', timeout=30) as response:
            health = json.loads(response.read())
        if (
            not health.get('loaded')
            or health.get('renderer') != cfg['renderer']
            or health.get('reward_hash') != digest(cfg['reward'])
        ):
            raise ValueError('Reward service is unavailable or differs from the experiment')
        if health.get('track_metrics', []) != cfg['track_metrics']:
            raise ValueError('Reward service track_metrics differ from the training configuration')
        services[url.rstrip('/')] = health['service_hash']
    identity['reward_services'] = services
    execution_identity = identity
    identity = stable_identity(identity)
    from omegaconf import open_dict

    with open_dict(composed):
        composed.reward.custom_reward_function.reward_kwargs.service_hashes = json.dumps(services)
    with run_lock(a.out):
        manifest = a.out / 'manifest.json'
        if a.resume:
            if not manifest.exists() or read_json(manifest) != identity:
                raise ValueError('Resume requires identical data, source, runtime, and settings')
        elif manifest.exists():
            raise ValueError('Run already exists; use --resume')
        atomic_json(manifest, identity)
        import time

        atomic_json(a.out / 'executions' / f'{time.time_ns()}.json', execution_identity)
        OmegaConf.save(composed, a.out / 'resolved-config.yaml')
        try:
            run_ppo(composed, TaskRunner)
        except BaseException as error:
            atomic_json(
                a.out / 'status.json',
                dict(state='failed', error=f'{type(error).__name__}: {error}'),
            )
            raise
        else:
            atomic_json(
                a.out / 'status.json',
                dict(
                    state='smoke_stopped' if a.stop_after_step else 'complete',
                    rows=audit['train_rows'],
                    epoch_steps=audit['steps'],
                    steps=(
                        min(audit['steps'], cfg['max_steps'])
                        if cfg['max_steps']
                        else audit['steps']
                    ),
                ),
            )


if __name__ == '__main__':
    main()
