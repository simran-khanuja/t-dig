"""Freeze one multi-node policy job per (T2I model, policy size). Preparation never submits.

Paths come from the environment: HF_HUB_CACHE (default $HF_HOME/hub) holds the Hugging Face
snapshots, T_DIG_MODELS the local scorer weights (dreamsim_cache, aesthetic head), and PY_TRAIN,
PY_REWARD and PY_VQA name the three interpreters (default: python on PATH). T_DIG_SUBMIT is the
submit command written into matrix.json (default: bash; e.g. "sbatch --nodes=2
--ntasks-per-node=1 --gpus-per-node=8 --time=72:00:00").
"""

import argparse
import copy
import json
import os
import shlex
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hpdv2.storage import atomic_json, read_json
from baselines.assets import file_hash, local_checkpoint
from elicitation.tpips_compat import TPIPS_CHECKPOINT
from axis_grpo.prepare import new_directory
from axis_grpo.dataset import export_parquet
from axis_grpo.training.config import settings
from axis_grpo.training.split import val_subset

# Qwen3.5-35B-A3B (70 GB bf16) needs rollout TP 2 and an offloaded FSDP actor to fit beside vLLM.
# Its routed experts are fused parameters, so the MLP LoRA targets reach the shared expert only.
SIZE_SETTINGS = {'35B-A3B': dict(rollout_tp=2, actor_offload=True)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument(
        '--hf-cache',
        type=Path,
        default=None,
        help='Hugging Face hub cache (default: $HF_HUB_CACHE, else $HF_HOME/hub)',
    )
    p.add_argument(
        '--models',
        type=Path,
        default=None,
        help='local scorer weights (default: $T_DIG_MODELS, else <repo>/models)',
    )
    p.add_argument('--tau', type=float, required=True)
    p.add_argument('--lam', type=float, required=True)
    p.add_argument('--learning-rate', type=float, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument(
        '--val-axes-per-prompt',
        type=int,
        default=None,
        help='validate on the N lowest-slot named axes of each held-out prompt plus its overall '
        'rows (default: every held-out row)',
    )
    p.add_argument(
        '--t2i-models',
        nargs='+',
        default=['sd35-medium', 'sd35-large', 'flux2-klein-9b', 'qwen-image-2512'],
        choices=['sd35-medium', 'sd35-large', 'flux2-klein-9b', 'qwen-image-2512'],
    )
    p.add_argument('--policy-observation', choices=['text', 'elicitation_images'], default='text')
    p.add_argument(
        '--metric',
        choices=['pairwise', 'vendi'],
        default='pairwise',
        help='diversity term of the reward',
    )
    p.add_argument(
        '--edit-strength',
        type=float,
        default=None,
        help='edit task, image-to-image editors (SD3.5) only: denoising strength; service '
        'default 0.6',
    )
    p.add_argument(
        '--edit-steps',
        type=int,
        default=None,
        help="edit task: editor denoising steps in training, in place of the generator's "
        'train steps; validation keeps the validation steps',
    )
    p.add_argument(
        '--editor-path',
        default=None,
        help='edit task, qwen-image-2512 only: Qwen-Image-Edit-2509 snapshot supplying the '
        "transformer and processor; the other components are the generator's. Default: "
        'the pinned revision in the hub cache',
    )
    p.add_argument(
        '--task',
        choices=['generate', 'edit'],
        default='generate',
        help='edit: the policy writes edit instructions for the k source images of each row; the '
        'reward adds a drift term and the length cap is that of the edit instruction',
    )
    p.add_argument(
        '--drift-weight', type=float, default=1.0, help='edit task: weight of the drift hinge'
    )
    p.add_argument(
        '--drift-threshold',
        type=float,
        default=0.35,
        help='edit task: DreamSim distance an edit may reach free of cost',
    )
    p.add_argument(
        '--lora',
        choices=['all', 'mixer'],
        default='all',
        help='all: attention, linear-attention and MLP projections (12 modules). '
        'mixer: attention and linear-attention projections only (9 modules), no MLP',
    )
    p.add_argument(
        '--nodes',
        type=int,
        default=2,
        help='8-GPU nodes per job: node 0 holds the 2 policy GPUs, a VQAScore GPU and 5 reward GPUs; '
        'each further node a VQAScore GPU and 7 reward GPUs (12 reward GPUs on 2 nodes, 19 on 3)',
    )
    p.add_argument(
        '--sizes',
        nargs='+',
        default=['4B', '9B'],
        choices=['4B', '9B', '35B-A3B'],
        help='Qwen3.5 policy sizes to prepare',
    )
    p.add_argument(
        '--quality-weight',
        type=float,
        default=0.0,
        help='reward cost per HPSv3 point by which an image falls below the quality_thresh of its row, '
        'less --quality-margin; 0 = no quality term. Needs hpsv3 in --track-metrics and a dataset '
        'with quality_thresh (tdig/iid_quality.py attach)',
    )
    p.add_argument(
        '--quality-margin',
        type=float,
        default=1.0,
        help='HPSv3 points below quality_thresh that cost nothing',
    )
    p.add_argument(
        '--length-cap',
        type=int,
        default=0,
        help='word cap of the length term; 0 = the cap of the T2I model family (60 SD3.5, 150 others, 30 edit). '
        'Set it to the cap the dataset instruction states (tdig/recap_dataset.py)',
    )
    p.add_argument(
        '--length-weight',
        type=float,
        default=0.0,
        help='reward cost of a prompt that is 100%% over the word cap of its instruction; 0 = no length term',
    )
    p.add_argument('--save-freq', type=int, default=100, help='optimizer steps between checkpoints')
    p.add_argument(
        '--save-freq-early',
        type=int,
        default=0,
        help='additionally save every this many updates until --save-freq-early-until',
    )
    p.add_argument('--save-freq-early-until', type=int, default=200)
    p.add_argument(
        '--max-steps',
        type=int,
        default=0,
        help='stop after this many optimizer steps; 0 trains one epoch',
    )
    p.add_argument(
        '--validation',
        choices=['inline', 'offline'],
        default='inline',
        help='offline: no validation inside training; score checkpoints in a separate job',
    )
    p.add_argument(
        '--track-metrics',
        nargs='*',
        default=[],
        choices=['imagereward', 'hpsv3', 'hpsv2', 'clip_score', 'aesthetic'],
        help='logged-only scores computed on every training and validation completion',
    )
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = a.out.resolve()
    hf_home = Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface'))
    cache = (a.hf_cache or Path(os.environ.get('HF_HUB_CACHE', hf_home / 'hub'))).resolve()
    models = (a.models or Path(os.environ.get('T_DIG_MODELS', root / 'models'))).resolve()
    dataset = a.dataset.resolve()
    python = {name: os.environ.get(name, 'python') for name in ('PY_TRAIN', 'PY_REWARD', 'PY_VQA')}
    import pyarrow.parquet as pq

    manifest = read_json(dataset / 'manifest.json')
    for name, expected in manifest['files'].items():
        if file_hash(dataset / name) != expected:
            raise ValueError(f'Dataset changed: {name}')
    specs = [
        json.loads(r['reward_model']['ground_truth'])
        for r in pq.read_table(dataset / 'train.parquet', columns=['reward_model']).to_pylist()
    ]
    per_row_tau = sum('align_thresh' in s for s in specs)
    if a.quality_weight and (
        sum('quality_thresh' in s for s in specs) != per_row_tau or 'hpsv3' not in a.track_metrics
    ):
        raise ValueError(
            '--quality-weight needs quality_thresh on every row and hpsv3 in --track-metrics'
        )
    policy = {}
    for size in a.sizes:
        repo = cache / f'models--Qwen--Qwen3.5-{size}'
        revision = (repo / 'refs/main').read_text().strip()
        path = repo / 'snapshots' / revision
        index = read_json(path / 'model.safetensors.index.json')
        for name in set(index['weight_map'].values()):
            if not (path / name).is_file():
                raise ValueError(f'Missing policy shard: {path/name}')
        policy[size] = str(path)
    reward = dict(
        metric=a.metric, mode='hinge', fidelity_weight=a.lam, fidelity_threshold=a.tau, floor='none'
    )
    if len(set(a.t2i_models)) != len(a.t2i_models):
        raise ValueError('Repeated T2I model')
    if a.task == 'edit':
        editors = {'flux2-klein-9b', 'sd35-medium', 'sd35-large', 'qwen-image-2512'}
        if (
            len(a.t2i_models) != 1
            or a.t2i_models[0] not in editors
            or a.policy_observation != 'elicitation_images'
            or manifest.get('task') != 'edit'
            or manifest.get('instruction_renderer') != a.t2i_models[0]
        ):
            raise ValueError(
                'The edit task needs one T2I model with an editor, --policy-observation elicitation_images '
                'and a dataset built by tdig.build_edit_dataset for that editor'
            )
        # qwen-image-2512 edits with the separate Qwen-Image-Edit-2509 checkpoint; the other
        # T2I models edit with the generator's own weights.
        editor_path = None
        if a.t2i_models[0] == 'qwen-image-2512':
            editor_path = (
                Path(a.editor_path)
                if a.editor_path
                else local_checkpoint(
                    'Qwen/Qwen-Image-Edit-2509@d3968ef930e841f4c73640fb8afa3b306a78167e', cache
                )
            )
            for name in (
                'transformer/diffusion_pytorch_model.safetensors.index.json',
                'processor/preprocessor_config.json',
            ):
                if not (editor_path / name).is_file():
                    raise ValueError(f'Missing Qwen-Image-Edit file: {editor_path/name}')
        elif a.editor_path:
            raise ValueError('--editor-path applies to the qwen-image-2512 editor only')
    elif manifest.get('task') == 'edit':
        raise ValueError('This dataset holds editing rows; pass --task edit')
    if (
        a.policy_observation == 'elicitation_images'
        and manifest.get('policy_observation') != 'elicitation_images'
    ):
        raise ValueError('Image-input jobs require the verified image-bearing export')
    pairs = [(t2i_model, size) for t2i_model in a.t2i_models for size in a.sizes]
    if a.nodes < 2:
        raise ValueError('--nodes must be at least 2')
    print(
        json.dumps(
            dict(
                execute=a.execute,
                jobs=len(pairs),
                nodes=a.nodes * len(pairs),
                gpus=8 * a.nodes * len(pairs),
                reward=reward,
                learning_rate=a.learning_rate,
            ),
            indent=2,
        )
    )
    if not a.execute:
        return
    with new_directory(out, (dataset, cache, models)) as staging:
        source = staging / 'source'
        source.mkdir()
        for module in ('axis_grpo', 'baselines', 'elicitation', 'hpdv2'):
            for path in (root / module).rglob('*'):
                if not path.is_file() or '__pycache__' in path.parts:
                    continue
                if path.suffix not in ('.py', '.json') and path.name not in (
                    'THIRD_PARTY.md',
                    'verl-LICENSE',
                ):
                    continue
                target = source / path.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        (source / 'scripts').mkdir()
        for name in ('env.sh', 'training_fleet.sh', 'training_fleet_node.sh'):
            shutil.copy2(root / 'scripts' / name, source / 'scripts' / name)
        shutil.copy2(Path(__file__), staging / 'prepare_training_matrix.py')
        frozen = {
            str(f.relative_to(source)): file_hash(f) for f in source.rglob('*') if f.is_file()
        }
        atomic_json(staging / 'source-hashes.json', frozen)
        # The first three training rows, unchanged, form every job's two-step (2+1 rows)
        # optimizer and resume qualification.
        train = pq.read_table(dataset / 'train.parquet').slice(0, 3).to_pylist()
        val = pq.read_table(dataset / 'val.parquet').to_pylist()
        export_parquet(staging / 'qualification-train.parquet', train)
        export_parquet(staging / 'qualification-val.parquet', val[:2])
        val_file = dataset / 'val.parquet'
        if a.val_axes_per_prompt:
            val = val_subset(val, a.val_axes_per_prompt)
            val_file = out / 'validation.parquet'
            export_parquet(staging / 'validation.parquet', val)
        jobs = []
        tpips = local_checkpoint(TPIPS_CHECKPOINT, cache)
        for t2i_model, size in pairs:
            # The cap tdig.instruction states to the policy for this T2I model. An SD3.5 editor is
            # image-to-image, so its edit instruction is a description, capped like a prompt.
            cap = (
                (60 if t2i_model.startswith('sd35') else 30)
                if a.task == 'edit'
                else 60 if t2i_model.startswith('sd35') else 150
            )
            if a.length_cap:
                cap = a.length_cap
            job_reward = (
                dict(reward, length_weight=a.length_weight, length_cap=cap)
                if a.length_weight
                else dict(reward)
            )
            if a.task == 'edit':
                job_reward.update(drift_weight=a.drift_weight, drift_threshold=a.drift_threshold)
            if a.quality_weight:
                job_reward.update(quality_weight=a.quality_weight, quality_margin=a.quality_margin)
            name = f'{t2i_model}-qwen35-{size.lower()}'
            run = out / name
            temp = staging / name
            temp.mkdir()
            asset = root / 'configs/t2i_models' / f'{t2i_model}.assets.lock.json'
            shutil.copy2(asset, temp / 'renderer-assets.json')
            if t2i_model.startswith('sd35'):
                steps, val_steps, guidance, maxlen = 16, 40, 4.5, 256
            elif t2i_model == 'flux2-klein-9b':
                steps, val_steps, guidance, maxlen = 4, 4, 1.0, 512
            else:
                steps, val_steps, guidance, maxlen = 16, 50, 4.0, 512
            if a.edit_steps is not None:
                if a.task != 'edit':
                    raise ValueError('--edit-steps applies to the edit task only')
                steps = a.edit_steps
            render = {
                split: dict(
                    resolution=res,
                    steps=s,
                    guidance=guidance,
                    max_sequence_length=maxlen,
                    negative_prompt='',
                )
                for split, res, s in [('train', 512, steps), ('val', 1024, val_steps)]
            }
            service = dict(
                renderer=t2i_model,
                renderer_assets=str(run / 'renderer-assets.json'),
                render=render,
                device='cuda:0',
                tpips_checkpoint=str(tpips),
                cache=str(cache),
                tpips_batch=4,
                render_batch=4,
                val_render_batch=1,
                staging=str(run / 'staging'),
                vqa_url='http://127.0.0.1:18740',
                vqa_model='clip-flant5-xxl',
                timeout=3600,
                reward=job_reward,
                batch_items=4,
                batch_wait_ms=30,
                save_images=False,
            )
            if a.task == 'edit':
                service.update(dreamsim_cache=str(models / 'dreamsim_cache'))
            if a.task == 'edit' and a.edit_strength is not None:
                service.update(edit_strength=a.edit_strength)
            if a.task == 'edit' and editor_path is not None:
                service.update(editor_path=str(editor_path))
            if a.track_metrics:
                service.update(
                    track_metrics=a.track_metrics,
                    aesthetic_head=str(models / 'aesthetic/sac_logos_ava1_l14_linearMSE.pth'),
                )
            atomic_json(temp / 'reward-config.json', service)
            train_cfg = settings(
                dict(
                    policy_path=policy[size],
                    train_file=str(dataset / 'train.parquet'),
                    val_file=str(val_file),
                    cache=str(cache),
                    renderer=t2i_model,
                    reward=job_reward,
                    reward_endpoints=['http://pending-allocation:18750'],
                    learning_rate=a.learning_rate,
                    train_gpus=2,
                    batch_size=8,
                    group_size=8,
                    policy_observation=a.policy_observation,
                    max_prompt_length=(
                        3072 if a.policy_observation == 'elicitation_images' else 1024
                    ),
                    save_freq=a.save_freq,
                    save_freq_early=a.save_freq_early,
                    save_freq_early_until=a.save_freq_early_until,
                    max_steps=a.max_steps,
                    validation=a.validation,
                    track_metrics=a.track_metrics,
                    test_freq=500 if a.validation == 'inline' else 0,
                    val_before_train=a.validation == 'inline',
                    val_rollouts=1,
                    reward_timeout=3600,
                    ray_cpus=32,
                    gdn_prefill_backend='triton',
                    allow_endpoint_rebind=True,
                    # Qwen3.5 has 8 full-attention layers (q/k/v/o) and 24 linear-attention layers
                    # (in_proj_*/out_proj); both groups are needed to adapt every layer's token mixer.
                    **SIZE_SETTINGS.get(size, {}),
                    lora_targets=[
                        'q_proj',
                        'k_proj',
                        'v_proj',
                        'o_proj',
                        'in_proj_qkv',
                        'in_proj_z',
                        'in_proj_a',
                        'in_proj_b',
                        'out_proj',
                    ]
                    + (['gate_proj', 'up_proj', 'down_proj'] if a.lora == 'all' else []),
                )
            )
            atomic_json(temp / 'train-config.json', train_cfg)
            smoke = copy.deepcopy(train_cfg)
            smoke.update(
                smoke=True,
                train_file=str(out / 'qualification-train.parquet'),
                val_file=str(out / 'qualification-val.parquet'),
                batch_size=2,
                save_freq=1,
                test_freq=0,
                val_before_train=False,
            )
            atomic_json(temp / 'qualification-config.json', smoke)
            fleet = dict(
                out=str(run),
                source=str(out / 'source'),
                cache=str(cache),
                train_python=python['PY_TRAIN'],
                reward_python=python['PY_REWARD'],
                vqa_python=python['PY_VQA'],
                reward_config=str(run / 'reward-config.json'),
                train_config=str(run / 'train-config.json'),
                qualification_config=str(run / 'qualification-config.json'),
                vqa_port=18740,
                reward_port=18750,
                vqa_batch=16,
                probe_timeout=7200,
                nodes=a.nodes,
            )
            atomic_json(temp / 'fleet-config.json', fleet)
            short = {
                'sd35-medium': 'sdm',
                'sd35-large': 'sdl',
                'flux2-klein-9b': 'flux',
                'qwen-image-2512': 'qimg',
            }[t2i_model]
            # One job script per run: it exports the run's settings and starts node zero. The submit
            # command (T_DIG_SUBMIT) carries the resource request: a.nodes nodes of 8 GPUs, one task
            # per node, 72 hours.
            (temp / 'job.sh').write_text(
                '#!/bin/bash\n'
                f'# grpo-{short}-{size.lower()}: {a.nodes} nodes x 8 GPUs, one task per node, 72 h.\n'
                f'export T_DIG_TRAIN_SOURCE={shlex.quote(str(out/"source"))}\n'
                f'export T_DIG_TRAIN_PYTHON={shlex.quote(python["PY_TRAIN"])}\n'
                f'export T_DIG_FLEET_CONFIG={shlex.quote(str(run/"fleet-config.json"))}\n'
                'exec /bin/bash "$T_DIG_TRAIN_SOURCE/scripts/training_fleet.sh"\n'
            )
            command = shlex.split(os.environ.get('T_DIG_SUBMIT', 'bash')) + [str(run / 'job.sh')]
            jobs.append(
                dict(
                    name=name,
                    renderer=t2i_model,
                    policy_size=size,
                    nodes=a.nodes,
                    gpus=8 * a.nodes,
                    config=str(run / 'fleet-config.json'),
                    submit_command=command,
                )
            )
        atomic_json(
            staging / 'matrix.json',
            dict(
                state='prepared_not_submitted',
                nodes=a.nodes * len(jobs),
                gpus=8 * a.nodes * len(jobs),
                jobs=jobs,
                dataset=str(dataset),
                dataset_manifest_sha256=file_hash(dataset / 'manifest.json'),
                val_file=str(val_file),
                val_rows=len(val),
                reward=reward,
                learning_rate=a.learning_rate,
            ),
        )
        (staging / 'README.md').write_text(
            f'# {len(jobs)} separate policy runs\n\n'
            f'Task: {a.task}. Policy observation: {a.policy_observation}. Policy sizes: Qwen3.5-{"/".join(a.sizes)}. Each allocation '
            f'has {a.nodes} 8-GPU nodes: 2 policy GPUs, {5+7*(a.nodes-1)} T2I model/TPIPS GPUs, and {a.nodes} fidelity GPUs.\n\n'
            f'Tau={a.tau} for rows without their own align_thresh, lambda={a.lam}, learning rate={a.learning_rate}; '
            f'{a.metric} TPIPS diversity, hinge fidelity penalty, no floor. '
            + (
                f'Length term: {a.length_weight} per 100% over the instruction word cap (30 for the edit task, else 60 SD3.5 and 150 others), capped at 300%. '
                if a.length_weight
                else 'No length term. '
            )
            + (
                f'Drift term: DreamSim, weight {a.drift_weight}, threshold {a.drift_threshold}. '
                if a.task == 'edit'
                else ''
            )
            + (
                f'Quality term: {a.quality_weight} per HPSv3 point below the row quality_thresh less {a.quality_margin}, capped at 6 points. '
                if a.quality_weight
                else ''
            )
            + (
                f'Image-to-image edit strength {a.edit_strength}. '
                if a.edit_strength is not None
                else ''
            )
            + (
                f'Editor trained at {a.edit_steps} denoising steps. '
                if a.edit_steps is not None
                else ''
            )
            + (
                f'Editor: Qwen-Image-Edit-2509 at {editor_path}, run at the train/val render settings of the generator. '
                if a.task == 'edit' and editor_path is not None
                else ''
            )
            + 'Rows with align_thresh: '
            f'{per_row_tau} of {manifest["rows"]}. '
            f'LoRA rank 32/alpha 64 on {"attention, linear-attention and MLP projections (12 modules)" if a.lora=="all" else "attention and linear-attention projections (9 modules, no MLP)"}. '
            f'{"One epoch" if not a.max_steps else f"At most {a.max_steps} optimizer steps"}, batch 8, '
            'eight GRPO completions per row, four expanded prompts per completion.\n\n'
            'Train at 512 px: SD3.5 uses 16 steps/CFG 4.5; distilled Klein 4 steps/CFG 1; Qwen 16 steps/CFG 4. '
            'Validation at 1024 px: SD3.5 40 steps, Klein 4 steps, Qwen 50 steps. '
            'Both resolutions are qualified for memory/scoring before training.\n\n'
            f'Checkpoints every {a.save_freq} updates'
            + (
                f', and every {a.save_freq_early} until update {a.save_freq_early_until}'
                if a.save_freq_early
                else ''
            )
            + '. '
            + (
                f'Validation on {len(val)} held-out rows before training and every 500 updates. '
                if a.validation == 'inline'
                else 'No validation inside training; checkpoints are scored by a separate job. '
            )
            + f'Logged-only scores on every completion: {", ".join(a.track_metrics) or "none"}. '
            + (
                f'Validation keeps {a.val_axes_per_prompt} named axes per held-out prompt plus its overall rows. '
                if a.val_axes_per_prompt
                else 'Validation uses every held-out row. '
            )
            + '\n\n'
            'Jobs first benchmark each node, then test actual policy gradients and checkpoint resume. '
            'Only successful qualification proceeds automatically to the full epoch. No images/API '
            'elicitation are rerun. T2I model outputs generated for rewards are not saved wholesale; '
            'per-completion reward receipts and policy rollouts are saved.\n\n'
            'matrix.json contains the exact submit commands. Preparation alone does not submit. '
            'A 72-hour allocation may need resubmission; complete checkpoints retain the curriculum cursor.\n'
        )


if __name__ == '__main__':
    main()
