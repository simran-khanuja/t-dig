"""One policy and one fixed reward T2I model per experiment; explicit run settings."""

from pathlib import Path
import json
import math
from baselines.config import T2I_MODELS
from .protocol import QUALITY_SCORER, validate_reward, validate_tracked

DEFAULTS = dict(
    policy_path=None,
    train_file=None,
    val_file=None,
    cache=None,
    smoke=False,
    renderer='sd35-medium',
    reward_endpoints=[],
    reward=dict(
        metric='pairwise', mode='hinge', fidelity_weight=0.3, fidelity_threshold=0.5, floor='none'
    ),
    train_gpus=2,
    batch_size=2,
    group_size=4,
    learning_rate=5e-6,
    max_prompt_length=1024,
    max_response_length=1024,
    lora_rank=32,
    lora_alpha=64,
    lora_targets=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    seed=42,
    render_seed=0,
    save_freq=25,
    save_freq_early=0,
    save_freq_early_until=200,
    test_freq=25,
    val_before_train=True,
    val_rollouts=2,
    rollout_memory_fraction=0.5,
    rollout_tp=1,
    token_batch=8192,
    actor_offload=False,
    reward_timeout=1800,
    ray_cpus=24,
    gdn_prefill_backend=None,
    allow_endpoint_rebind=False,
    policy_observation='text',
    input_images=4,
    track_metrics=[],
    validation='inline',
    max_steps=0,
)


def settings(value):
    unknown = set(value) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown training settings: {sorted(unknown)}')
    cfg = {**DEFAULTS, **value}
    validate_reward(cfg['reward'])
    if not isinstance(cfg['renderer'], str) or not cfg['renderer']:
        raise ValueError('Name the fixed reward T2I model')
    for key in ('policy_path', 'train_file', 'val_file', 'cache'):
        if not cfg[key] or not Path(cfg[key]).exists():
            raise ValueError(f'Missing local {key}')
        cfg[key] = str(Path(cfg[key]).resolve())
    if not (Path(cfg['policy_path']) / 'config.json').is_file():
        raise ValueError('Use a pinned local policy snapshot, not a moving model alias')
    for key in (
        'train_gpus',
        'batch_size',
        'group_size',
        'lora_rank',
        'lora_alpha',
        'max_prompt_length',
        'max_response_length',
        'save_freq',
        'val_rollouts',
        'rollout_tp',
        'token_batch',
        'ray_cpus',
    ):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'Invalid {key}')
    for key in ('save_freq_early', 'save_freq_early_until'):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError(f'Invalid {key}')
    for key in ('smoke', 'val_before_train', 'actor_offload', 'allow_endpoint_rebind'):
        if type(cfg[key]) is not bool:
            raise ValueError(f'{key} must be boolean')
    # max_steps=0 trains the whole epoch; a positive value stops after that many optimizer steps.
    for key in ('seed', 'render_seed', 'test_freq', 'max_steps'):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError(f'Invalid {key}')
    for key in ('learning_rate', 'reward_timeout'):
        if not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'Invalid {key}')
    if (
        not isinstance(cfg['lora_targets'], list)
        or not cfg['lora_targets']
        or not all(isinstance(v, str) and v for v in cfg['lora_targets'])
    ):
        raise ValueError('Require explicit LoRA target module names')
    if cfg['group_size'] < 2 or cfg['group_size'] % cfg['train_gpus']:
        raise ValueError(
            'Group size must be divisible by trainer GPU count, including short batches'
        )
    if cfg['train_gpus'] % cfg['rollout_tp'] or not 0 < cfg['rollout_memory_fraction'] < 1:
        raise ValueError('Invalid rollout parallelism or GPU memory fraction')
    if not cfg['reward_endpoints'] or not all(
        isinstance(u, str) and u.startswith('http://') for u in cfg['reward_endpoints']
    ):
        raise ValueError('Configure explicit internal reward service URLs')
    validate_tracked(cfg['track_metrics'])
    # validation='inline' validates inside the training loop; 'offline' leaves checkpoints to a
    # separate scoring job.
    if cfg['validation'] not in ('inline', 'offline'):
        raise ValueError("validation must be 'inline' or 'offline'")
    if cfg['validation'] == 'offline' and (cfg['test_freq'] != 0 or cfg['val_before_train']):
        raise ValueError('Offline validation requires test_freq=0 and val_before_train=false')
    if (
        not cfg['smoke']
        and cfg['validation'] == 'inline'
        and (cfg['test_freq'] < 1 or not cfg['val_before_train'])
    ):
        raise ValueError(
            'Inline validation requires held-out validation before and during training'
        )
    if cfg['gdn_prefill_backend'] not in (None, 'triton'):
        raise ValueError('Unsupported GDN prefill backend')
    if cfg['policy_observation'] not in ('text', 'elicitation_images') or cfg['input_images'] != 4:
        raise ValueError('Require text or four original elicitation images')
    return cfg


def validate_service(cfg):
    required = {
        'renderer',
        'renderer_assets',
        'render',
        'device',
        'tpips_checkpoint',
        'cache',
        'tpips_batch',
        'render_batch',
        'staging',
        'vqa_url',
        'vqa_model',
        'timeout',
        'reward',
        'batch_items',
        'batch_wait_ms',
        'save_images',
    }
    optional = {
        'val_render_batch',
        'track_metrics',
        'aesthetic_head',
        'dreamsim_cache',
        'edit_strength',
        'editor_path',
        'pipeline',
    }
    missing, unknown = required - set(cfg), set(cfg) - required - optional
    if missing or unknown:
        raise ValueError(
            f'Reward service keys: missing {sorted(missing)}, unknown {sorted(unknown)}'
        )
    if 'val_render_batch' in cfg and (
        type(cfg['val_render_batch']) is not int or cfg['val_render_batch'] < 1
    ):
        raise ValueError('Invalid val_render_batch')
    # A name outside T2I_MODELS is a custom Diffusers text-to-image model; `pipeline` names its
    # pipeline class (default AutoPipelineForText2Image). Custom models generate only.
    if not isinstance(cfg['renderer'], str) or not cfg['renderer']:
        raise ValueError('Name the T2I model')
    custom = cfg['renderer'] not in T2I_MODELS
    if 'pipeline' in cfg and (
        not custom or not isinstance(cfg['pipeline'], str) or not cfg['pipeline']
    ):
        raise ValueError('pipeline names the Diffusers pipeline class of a custom T2I model')
    if custom and 'drift_weight' in cfg['reward']:
        raise ValueError('Image editing is wired for the four paper T2I models only')
    if 'drift_weight' in cfg['reward'] and not Path(cfg.get('dreamsim_cache') or '').is_dir():
        raise ValueError(
            'A reward with a drift term needs a T2I model with an editor and a dreamsim_cache directory'
        )
    # editor_path applies to qwen-image-2512 only: its editor is the separate Qwen-Image-Edit-2509
    # checkpoint, while the other T2I models edit with their own weights.
    if (
        'drift_weight' in cfg['reward']
        and cfg['renderer'] == 'qwen-image-2512'
        and not Path(cfg.get('editor_path') or '').is_dir()
    ):
        raise ValueError(
            'The qwen-image-2512 editor needs editor_path, the Qwen-Image-Edit-2509 snapshot'
        )
    if 'editor_path' in cfg and (
        cfg['renderer'] != 'qwen-image-2512' or 'drift_weight' not in cfg['reward']
    ):
        raise ValueError('editor_path applies to the qwen-image-2512 edit task only')
    tracked = validate_tracked(cfg.get('track_metrics', []))
    if 'quality_weight' in cfg['reward'] and QUALITY_SCORER not in tracked:
        raise ValueError(f'A reward with a quality term needs {QUALITY_SCORER} in track_metrics')
    if 'aesthetic' in tracked and not Path(cfg.get('aesthetic_head') or '').is_file():
        raise ValueError(
            'track_metrics aesthetic requires aesthetic_head, the LAION linear head file'
        )
    validate_reward(cfg['reward'])
    if set(cfg['render']) != {'train', 'val'}:
        raise ValueError('Require explicit train and validation render settings')
    for render in cfg['render'].values():
        if set(render) != {
            'resolution',
            'steps',
            'guidance',
            'max_sequence_length',
            'negative_prompt',
        }:
            raise ValueError('Unknown/missing rendering setting')
        for key in ['resolution', 'steps', 'max_sequence_length']:
            if type(render[key]) is not int or render[key] < 1:
                raise ValueError(f'Invalid render {key}')
        if render['resolution'] % 64:
            raise ValueError('Use resolution divisible by 64 for the supported native pipelines')
        if not isinstance(render['negative_prompt'], str):
            raise ValueError('Negative prompt must be text')
        if (
            not isinstance(render['guidance'], (int, float))
            or not math.isfinite(render['guidance'])
            or render['guidance'] < 0
        ):
            raise ValueError('Invalid guidance')
        if cfg['renderer'] == 'flux2-klein-9b' and render['guidance'] != 1:
            raise ValueError('Distilled Klein uses guidance 1')
    for key in ['tpips_batch', 'render_batch', 'batch_items', 'timeout']:
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'Invalid {key}')
    if type(cfg['batch_wait_ms']) is not int or cfg['batch_wait_ms'] < 0:
        raise ValueError('Invalid batch wait')
    if type(cfg['save_images']) is not bool:
        raise ValueError('save_images must be boolean')
    if cfg['vqa_model'] not in {'clip-flant5-xxl', 'clip-flant5-xl'}:
        raise ValueError('Unsupported fidelity model')
    if not cfg['vqa_url'].startswith('http://'):
        raise ValueError('Use an internal VQA endpoint')
    return cfg


def compose_config(cfg, out, resume=False, stop_after_step=0):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import verl

    with initialize_config_dir(
        config_dir=str(Path(verl.__file__).parent / 'trainer/config'), version_base=None
    ):
        result = compose(config_name='ppo_trainer')
    a = 'actor_rollout_ref.'
    values = {
        'trainer.use_v1': True,
        'trainer.v1.trainer_mode': 'sync',
        'trainer.total_epochs': 1,
        'trainer.total_training_steps': None,
        'trainer.n_gpus_per_node': cfg['train_gpus'],
        'trainer.nnodes': 1,
        'trainer.default_local_dir': str(Path(out).resolve() / 'checkpoints'),
        'trainer.project_name': 'tdig-axis-grpo',
        'trainer.experiment_name': Path(out).name,
        'trainer.logger': ['console'],
        'trainer.save_freq': cfg['save_freq'],
        'trainer.test_freq': cfg['test_freq'],
        'trainer.val_before_train': cfg['val_before_train'],
        'trainer.resume_mode': 'auto' if resume else 'disable',
        'trainer.rollout_data_dir': str(Path(out).resolve() / 'rollouts'),
        'trainer.validation_data_dir': str(Path(out).resolve() / 'validation'),
        'algorithm.adv_estimator': 'grpo',
        'algorithm.use_kl_in_reward': False,
        'critic.enable': False,
        'data.train_files': [cfg['train_file']],
        'data.val_files': [cfg['val_file']],
        'data.train_batch_size': cfg['batch_size'],
        'data.val_batch_size': None,
        'data.shuffle': False,
        'data.validation_shuffle': False,
        'data.filter_overlong_prompts': False,
        'data.train_max_samples': -1,
        'data.val_max_samples': -1,
        'data.truncation': 'error',
        'data.dataloader_num_workers': 0,
        'data.max_prompt_length': cfg['max_prompt_length'],
        'data.max_response_length': cfg['max_response_length'],
        'data.apply_chat_template_kwargs.enable_thinking': False,
        a + 'model.path': cfg['policy_path'],
        a + 'model.enable_gradient_checkpointing': True,
        # The Qwen3.5 torch fallback kernels (recurrent and causal convolution) ignore packed
        # sequence boundaries, so packing stays off. Re-enable it only after a model-level
        # isolation test.
        a + 'model.use_remove_padding': False,
        a + 'model.override_config.attn_implementation': 'sdpa',
        a + 'model.lora_rank': cfg['lora_rank'],
        a + 'model.lora_alpha': cfg['lora_alpha'],
        a + 'model.target_modules': cfg['lora_targets'],
        a + 'actor.strategy': 'fsdp2',
        a + 'actor.ppo_epochs': 1,
        a + 'actor.ppo_mini_batch_size': cfg['batch_size'],
        a + 'actor.use_dynamic_bsz': True,
        a + 'actor.ppo_max_token_len_per_gpu': cfg['token_batch'],
        a + 'actor.optim.lr': cfg['learning_rate'],
        a + 'actor.use_kl_loss': False,
        a + 'actor.entropy_coeff': 0.0,
        a + 'actor.fsdp_config.model_dtype': 'bfloat16',
        a + 'actor.fsdp_config.seed': cfg['seed'],
        a + 'actor.fsdp_config.param_offload': cfg['actor_offload'],
        a + 'actor.fsdp_config.optimizer_offload': cfg['actor_offload'],
        a + 'actor.checkpoint.async_save': False,
        a + 'rollout.name': 'vllm',
        a + 'rollout.load_format': 'safetensors',
        a + 'rollout.tensor_model_parallel_size': cfg['rollout_tp'],
        a + 'rollout.n': cfg['group_size'],
        a + 'rollout.temperature': 1.0,
        a + 'rollout.gpu_memory_utilization': cfg['rollout_memory_fraction'],
        a + 'rollout.max_model_len': cfg['max_prompt_length'] + cfg['max_response_length'],
        a + 'rollout.max_num_seqs': max(cfg['batch_size'] * cfg['group_size'], 16),
        a + 'rollout.engine_kwargs.vllm.disable_custom_all_reduce': True,
        a + 'rollout.log_prob_use_dynamic_bsz': True,
        a + 'rollout.log_prob_max_token_len_per_gpu': cfg['token_batch'],
        a + 'rollout.val_kwargs.n': cfg['val_rollouts'],
        a + 'rollout.val_kwargs.do_sample': True,
        a + 'rollout.val_kwargs.temperature': 1.0,
        a + 'ref.log_prob_use_dynamic_bsz': True,
        a + 'ref.log_prob_max_token_len_per_gpu': cfg['token_batch'],
        'reward.reward_manager.source': 'register',
        'reward.reward_manager.name': 'naive',
        'reward.custom_reward_function.path': str(
            Path(__file__).with_name('reward_fn.py').resolve()
        ),
        'reward.custom_reward_function.name': 'compute_score',
        'reward.custom_reward_function.reward_kwargs': dict(
            routes=json.dumps({cfg['renderer']: cfg['reward_endpoints']}),
            reward=json.dumps(cfg['reward']),
            renderer=cfg['renderer'],
            seed=cfg['render_seed'],
            timeout=cfg['reward_timeout'],
            tracked=json.dumps(cfg['track_metrics']),
        ),
        'transfer_queue.enable': True,
        'ray_kwargs.ray_init.num_cpus': cfg['ray_cpus'],
        'tdig': dict(
            stop_after_step=stop_after_step,
            max_steps=cfg['max_steps'],
            save_freq_early=cfg['save_freq_early'],
            save_freq_early_until=cfg['save_freq_early_until'],
        ),
    }
    for key, value in values.items():
        OmegaConf.update(result, key, value, merge=False, force_add=True)
    if cfg['gdn_prefill_backend']:
        OmegaConf.update(
            result,
            a + 'rollout.engine_kwargs.vllm.gdn_prefill_backend',
            cfg['gdn_prefill_backend'],
            merge=False,
            force_add=True,
        )
    if cfg['policy_observation'] == 'elicitation_images':
        OmegaConf.update(result, a + 'rollout.limit_images', cfg['input_images'], force_add=True)
    return result
