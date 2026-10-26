"""Train one T-DIG policy on one machine, with any number of GPUs, against any T2I model.

The script writes the run's configurations, downloads missing weights and data, starts the
VQAScore and reward services, qualifies the T2I model, trains the policy with GRPO and merges
the last checkpoint into a PEFT LoRA adapter, `<out>/adapter/lora_adapter`. Without --execute
it prints the plan. Paths and interpreters come from the environment (scripts/env.sh).

The T2I model is one of the paper's four (`sd35-medium`, `sd35-large`, `flux2-klein-9b`,
`qwen-image-2512`) or a directory holding any Diffusers text-to-image model, whose pipeline
class `--pipeline` names. The paper's policies train on the released rows of their T2I model
(neulab/t-dig-train); a custom model trains on the same rows without the per-row thresholds,
which were measured on the paper's models, so `--tau` sets the fidelity threshold of every row.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hpdv2.storage import atomic_json, read_json, scheduler_job_id
from baselines.assets import file_hash, local_checkpoint, lock_assets
from baselines.config import T2I_MODELS
from elicitation.tpips_compat import TPIPS_CHECKPOINT
from axis_grpo.dataset import export_parquet
from axis_grpo.training.config import settings
from axis_grpo.training.fleet import Processes, serve_node, single_node_roles

POLICIES = {
    '2B': 'Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc',
    '4B': 'Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
    '9B': 'Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a',
    '35B-A3B': 'Qwen/Qwen3.5-35B-A3B@59d61f3ce65a6d9863b86d2e96597125219dc754',
}
# Qwen3.5-35B-A3B (70 GB bf16) needs rollout TP 2 and an offloaded FSDP actor beside vLLM.
SIZE_SETTINGS = {'35B-A3B': dict(rollout_tp=2, actor_offload=True)}
T2I_REPOS = {
    'sd35-large': 'stabilityai/stable-diffusion-3.5-large@ceddf0a7fdf2064ea28e2213e3b84e4afa170a0f',
    'sd35-medium': (
        'stabilityai/stable-diffusion-3.5-medium@b940f670f0eda2d07fbb75229e779da1ad11eb80'
    ),
    'flux2-klein-9b': 'black-forest-labs/FLUX.2-klein-9B@92196c8e11f7b6cf2b7493e037d8c5345c559216',
    'qwen-image-2512': 'Qwen/Qwen-Image-2512@25468b98e3276ca6700de15c6628e51b7de54a26',
}
VQA_REPOS = {
    'clip-flant5-xxl': ['zhiqiulin/clip-flant5-xxl', 'google/flan-t5-xxl'],
    'clip-flant5-xl': ['zhiqiulin/clip-flant5-xl', 'google/flan-t5-xl'],
}
# Training steps, validation steps, guidance and text length of the paper's T2I models.
RENDER = {
    'sd35-medium': (16, 40, 4.5, 256),
    'sd35-large': (16, 40, 4.5, 256),
    'flux2-klein-9b': (4, 4, 1.0, 512),
    'qwen-image-2512': (16, 50, 4.0, 512),
}
# Qwen3.5 has 8 full-attention layers (q/k/v/o) and 24 linear-attention layers (in_proj_*/out_proj).
LORA_TARGETS = [
    'q_proj',
    'k_proj',
    'v_proj',
    'o_proj',
    'in_proj_qkv',
    'in_proj_z',
    'in_proj_a',
    'in_proj_b',
    'out_proj',
    'gate_proj',
    'up_proj',
    'down_proj',
]
DATASET_REPO = 'neulab/t-dig-train'
TEMPLATE = 'flux2-klein-9b'  # released rows a custom T2I model trains on


def gpus():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        return visible.split(',')
    return subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], text=True
    ).split()


def gpu_memory(gpu):
    """Return (total, free) memory of one GPU in MiB."""
    line = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.total,memory.free', '--format=csv,noheader,nounits']
        + ['-i', gpu],
        text=True,
    )
    return tuple(int(v) for v in line.split(','))


def download(repo, cache=None, **kwargs):
    from huggingface_hub import snapshot_download

    name, _, revision = repo.partition('@')
    return Path(snapshot_download(name, revision=revision or None, cache_dir=cache, **kwargs))


def released_dataset(data, name):
    directory = data / 't-dig-train' / 'data' / name
    if not (directory / 'manifest.json').is_file():
        print(f'downloading {DATASET_REPO} {name} to {directory}')
        download(
            DATASET_REPO,
            repo_type='dataset',
            local_dir=data / 't-dig-train',
            allow_patterns=[f'data/{name}/*'],
        )
    return directory


def derive_dataset(template, out, renderer):
    """Copy the template rows to `out` without their per-row fidelity and quality thresholds."""
    import pyarrow.parquet as pq

    out.mkdir(parents=True, exist_ok=True)
    manifest = read_json(template / 'manifest.json')
    for name in ('train.parquet', 'val.parquet'):
        if file_hash(template / name) != manifest['files'][name]:
            raise ValueError(f'Dataset changed: {template / name}')
        rows = pq.read_table(template / name).to_pylist()
        for row in rows:
            spec = json.loads(row['reward_model']['ground_truth'])
            spec.pop('align_thresh', None)
            spec.pop('quality_thresh', None)
            row['reward_model']['ground_truth'] = json.dumps(spec, ensure_ascii=False)
            row['extra_info']['instruction_renderer'] = renderer
        export_parquet(out / name, rows)
    manifest.update(
        instruction_renderer=renderer,
        source_dataset=str(template),
        thresholds='dropped: align_thresh and quality_thresh were measured on another T2I model',
    )
    manifest['files'] = {f.name: file_hash(f) for f in sorted(out.iterdir()) if f.is_file()}
    atomic_json(out / 'manifest.json', manifest)
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        '--t2i-model',
        required=True,
        help='one of ' + ', '.join(sorted(T2I_MODELS)) + ', or a Diffusers model directory',
    )
    p.add_argument(
        '--pipeline',
        default=None,
        help='custom model: its Diffusers pipeline class (default AutoPipelineForText2Image)',
    )
    p.add_argument('--out', type=Path, required=True, help='new run directory')
    p.add_argument(
        '--policy',
        default='4B',
        help='Qwen3.5 size (' + ', '.join(POLICIES) + ') or a local snapshot directory',
    )
    p.add_argument(
        '--dataset',
        type=Path,
        default=None,
        help="directory with train.parquet, val.parquet and manifest.json (default: the T2I "
        f"model's released rows, or for a custom model the {TEMPLATE} rows without thresholds)",
    )
    r = p.add_argument_group('render settings of a custom T2I model')
    r.add_argument('--steps', type=int, default=28)
    r.add_argument('--guidance', type=float, default=4.5)
    r.add_argument('--resolution', type=int, default=512, help='training images; multiple of 64')
    r.add_argument('--val-resolution', type=int, default=1024)
    r.add_argument('--max-sequence-length', type=int, default=256)
    g = p.add_argument_group('reward and optimization (defaults: the paper)')
    g.add_argument(
        '--tau', type=float, default=0.9, help='fidelity threshold of rows without their own'
    )
    g.add_argument('--lam', type=float, default=2.0, help='fidelity hinge weight')
    g.add_argument('--metric', choices=['pairwise', 'vendi'], default='vendi')
    g.add_argument('--length-weight', type=float, default=0.1)
    g.add_argument(
        '--quality-weight',
        type=float,
        default=0.0,
        help='HPSv3 quality hinge (paper: 0.05); loads a 17 GB scorer on every reward GPU and '
        'needs rows with quality_thresh',
    )
    g.add_argument('--quality-margin', type=float, default=0.0)
    g.add_argument('--learning-rate', type=float, default=2e-5)
    g.add_argument('--batch-size', type=int, default=8, help='rows per optimizer step')
    g.add_argument('--group-size', type=int, default=8, help='GRPO completions per row')
    g.add_argument('--max-steps', type=int, default=2000, help='0 trains one epoch')
    g.add_argument('--save-freq', type=int, default=50)
    g.add_argument(
        '--track-metrics',
        nargs='*',
        default=[],
        choices=['imagereward', 'hpsv3', 'hpsv2', 'clip_score', 'aesthetic'],
        help='logged-only image scores; --quality-weight needs hpsv3',
    )
    m = p.add_argument_group('GPUs and memory')
    m.add_argument(
        '--train-gpus', type=int, default=None, help='policy GPUs (default 2 of 8+, else 1)'
    )
    m.add_argument(
        '--rollout-memory-fraction',
        type=float,
        default=None,
        help="vLLM's share of the policy GPU (default: half of what is free when training starts)",
    )
    m.add_argument(
        '--token-batch',
        type=int,
        default=8192,
        help='tokens per policy forward pass; lower it on small GPUs',
    )
    m.add_argument(
        '--vqa-model',
        choices=list(VQA_REPOS),
        default=None,
        help='VQAScore model (default: xxl on a dedicated GPU of 40 GB or more, else xl)',
    )
    m.add_argument('--port', type=int, default=18740)
    p.add_argument('--resume', action='store_true', help='continue an interrupted run')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    data = Path(os.environ.get('T_DIG_DATA', root / 'data')).resolve()
    models = Path(os.environ.get('T_DIG_MODELS', root / 'models')).resolve()
    hf_home = Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface'))
    cache = Path(os.environ.get('HF_HUB_CACHE', hf_home / 'hub')).resolve()
    python = {name: os.environ.get(name, 'python') for name in ('PY_TRAIN', 'PY_REWARD', 'PY_VQA')}
    out = a.out.resolve()
    if out.exists() != a.resume:
        raise ValueError('Use a new --out directory, or --resume an existing run')

    visible = gpus()
    roles = single_node_roles(visible, a.train_gpus)
    shared = len(visible) <= 2
    custom = a.t2i_model not in T2I_MODELS
    renderer = Path(a.t2i_model).name if custom else a.t2i_model
    if custom and (renderer in T2I_MODELS or not Path(a.t2i_model).is_dir()):
        raise ValueError('A custom T2I model is a directory, named unlike the paper models')
    if a.pipeline and not custom:
        raise ValueError('--pipeline applies to a custom T2I model')
    if a.quality_weight and 'hpsv3' not in a.track_metrics:
        raise ValueError('--quality-weight needs hpsv3 in --track-metrics')
    vqa_total = gpu_memory(roles['vqa'])[0]
    vqa_model = a.vqa_model or (
        'clip-flant5-xxl' if not shared and vqa_total >= 40 * 1024 else 'clip-flant5-xl'
    )
    if custom:
        steps, val_steps, guidance, maxlen = a.steps, a.steps, a.guidance, a.max_sequence_length
    else:
        steps, val_steps, guidance, maxlen = RENDER[a.t2i_model]
    render = {
        split: dict(
            resolution=res,
            steps=s,
            guidance=guidance,
            max_sequence_length=maxlen,
            negative_prompt='',
        )
        for split, res, s in [('train', a.resolution, steps), ('val', a.val_resolution, val_steps)]
    }
    reward = dict(
        metric=a.metric, mode='hinge', fidelity_weight=a.lam, fidelity_threshold=a.tau, floor='none'
    )
    if a.length_weight:
        reward.update(length_weight=a.length_weight, length_cap=150)
    if a.quality_weight:
        reward.update(quality_weight=a.quality_weight, quality_margin=a.quality_margin)
    if a.dataset:
        rows = str(a.dataset)
    else:
        rows = f'{TEMPLATE} rows without thresholds' if custom else 'released rows'
    plan = dict(
        gpus=visible,
        roles=roles,
        t2i_model=a.t2i_model,
        renderer=renderer,
        policy=a.policy,
        dataset=rows,
        vqa_model=vqa_model,
        render=render,
        reward=reward,
        batch_size=a.batch_size,
        group_size=a.group_size,
        images_per_step=a.batch_size * a.group_size * 4,
        max_steps=a.max_steps,
        out=str(out),
    )
    print(json.dumps(dict(execute=a.execute, plan=plan), indent=2), flush=True)
    if not a.execute:
        return

    # Weights and data; everything after this point runs offline.
    policy = local_checkpoint(POLICIES.get(a.policy, a.policy), cache)
    if not (policy / 'config.json').is_file():
        raise ValueError(f'Not a policy snapshot: {policy}')
    tpips = local_checkpoint(TPIPS_CHECKPOINT, cache)
    for repo in VQA_REPOS[vqa_model] + ['openai/clip-vit-large-patch14-336']:
        download(repo, cache)
    if custom:
        t2i_path = Path(a.t2i_model).resolve()
    else:
        t2i_path = models / a.t2i_model
        if not t2i_path.is_dir():
            print(f'downloading {T2I_REPOS[a.t2i_model]} to {t2i_path}', flush=True)
            download(T2I_REPOS[a.t2i_model], local_dir=t2i_path)
    if a.dataset:
        dataset = a.dataset.resolve()
    elif custom:
        dataset = out / 'dataset'
        if not (dataset / 'manifest.json').is_file():
            derive_dataset(released_dataset(data, TEMPLATE), dataset, renderer)
    else:
        dataset = released_dataset(data, a.t2i_model)

    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / 'plan.json', plan)
    assets = out / 'renderer-assets.json'
    if not assets.exists():
        if custom:
            print(f'hashing {t2i_path}', flush=True)
            lock_assets({'renderer': str(t2i_path)}, assets)
        else:
            shutil.copy2(root / 'configs/t2i_models' / f'{a.t2i_model}.assets.lock.json', assets)
    service = dict(
        renderer=renderer,
        renderer_assets=str(assets),
        render=render,
        device='cuda:0',
        tpips_checkpoint=str(tpips),
        cache=str(cache),
        tpips_batch=4,
        render_batch=4,
        val_render_batch=1,
        staging=str(out / 'staging'),
        vqa_url=f'http://127.0.0.1:{a.port}',
        vqa_model=vqa_model,
        timeout=3600,
        reward=reward,
        batch_items=4,
        batch_wait_ms=30,
        save_images=False,
    )
    if a.pipeline:
        service['pipeline'] = a.pipeline
    if a.track_metrics:
        service.update(
            track_metrics=a.track_metrics,
            aesthetic_head=str(models / 'aesthetic/sac_logos_ava1_l14_linearMSE.pth'),
        )
    if not (out / 'reward-config.json').exists():
        atomic_json(out / 'reward-config.json', service)
    fleet = dict(
        out=str(out),
        cache=str(cache),
        train_python=python['PY_TRAIN'],
        reward_python=python['PY_REWARD'],
        vqa_python=python['PY_VQA'],
        reward_config=str(out / 'reward-config.json'),
        vqa_port=a.port,
        reward_port=a.port + 10,
        vqa_batch=16,
        probe_timeout=7200,
    )
    session = scheduler_job_id() or f'local-{os.getpid()}'
    os.environ['T_DIG_JOB_ID'] = session
    children = Processes(out, fleet)
    local = None
    # A signal stops the services too: it raises, and the `finally` closes every child.
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, lambda signum, frame: sys.exit(f'Received signal {signum}'))
    try:
        local, _, urls = serve_node(fleet, 0, session, roles)
        config_path = out / 'train-config.json'
        if config_path.exists():
            config = read_json(config_path)
            config['reward_endpoints'] = urls
        else:
            fraction = a.rollout_memory_fraction
            if fraction is None:
                total, free = min(gpu_memory(g) for g in roles['trainer'])
                fraction = max(0.1, round(0.5 * free / total, 2))
            config = settings(
                dict(
                    policy_path=str(policy),
                    train_file=str(dataset / 'train.parquet'),
                    val_file=str(dataset / 'val.parquet'),
                    cache=str(cache),
                    renderer=renderer,
                    reward=reward,
                    reward_endpoints=urls,
                    learning_rate=a.learning_rate,
                    train_gpus=len(roles['trainer']),
                    batch_size=a.batch_size,
                    group_size=a.group_size,
                    save_freq=a.save_freq,
                    max_steps=a.max_steps,
                    validation='offline',
                    test_freq=0,
                    val_before_train=False,
                    val_rollouts=1,
                    track_metrics=a.track_metrics,
                    reward_timeout=3600,
                    ray_cpus=min(32, os.cpu_count() or 32),
                    rollout_memory_fraction=fraction,
                    token_batch=a.token_batch,
                    gdn_prefill_backend='triton',
                    allow_endpoint_rebind=True,
                    lora_targets=LORA_TARGETS,
                    **SIZE_SETTINGS.get(a.policy, {}),
                )
            )
            atomic_json(config_path, config)
        resolved = out / 'resolved-train-input.json'
        atomic_json(resolved, config)
        command = [python['PY_TRAIN'], '-m', 'axis_grpo.training.run', '--config', str(resolved)]
        command += ['--out', str(out / 'policy'), '--execute']
        if (out / 'policy/manifest.json').exists():
            command.append('--resume')
        print(json.dumps(dict(training=True, endpoints=urls, log=str(out / 'train.log'))))
        trainer = children.start('train', command, roles['trainer'])
        while trainer.poll() is None:
            local.check()
            time.sleep(5)
        if trainer.returncode:
            raise RuntimeError(f'Training exited {trainer.returncode}; see {out / "train.log"}')
    finally:
        if local:
            local.close()
        children.close()
    checkpoints = out / 'policy' / 'checkpoints'
    steps = sorted(
        int(c.name.split('_')[-1])
        for c in checkpoints.glob('global_step_*')
        if (c / 'tdig-complete.json').is_file()
    )
    if not steps:
        raise RuntimeError(f'No complete checkpoint under {checkpoints}')
    actor = checkpoints / f'global_step_{steps[-1]}' / 'actor'
    merge = [python['PY_TRAIN'], '-m', 'verl.model_merger', 'merge', '--backend', 'fsdp']
    merge += ['--local_dir', str(actor), '--target_dir', str(out / 'adapter')]
    subprocess.run(merge + ['--use_cpu_initialization'], check=True)
    adapter = out / 'adapter' / 'lora_adapter'
    if not (adapter / 'adapter_config.json').is_file():
        raise RuntimeError(f'model_merger wrote no adapter under {out / "adapter"}')
    print(json.dumps(dict(complete=True, step=steps[-1], adapter=str(adapter))))


if __name__ == '__main__':
    main()
