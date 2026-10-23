"""Pass-bounded batches: consume each exported row once, including short tails."""

from hpdv2.storage import digest


class PhaseBatchSampler:
    def __init__(self, metadata, batch_size):
        if type(batch_size) is not int or batch_size < 1 or not metadata:
            raise ValueError('Require rows and a positive batch size')
        self.batches = []
        self.metadata = metadata
        last_pass, pending, seen = 0, [], set()
        for i, row in enumerate(metadata):
            phase = row['pass_number']
            if (
                row['index'] != i
                or type(phase) is not int
                or not 1 <= phase <= 4
                or phase < last_pass
            ):
                raise ValueError('Dataset indices/passes must preserve the exported order')
            if row['row_id'] in seen:
                raise ValueError('Duplicate training row')
            seen.add(row['row_id'])
            if pending and (phase != last_pass or len(pending) == batch_size):
                self.batches.append(pending)
                pending = []
            pending.append(i)
            last_pass = phase
        if pending:
            self.batches.append(pending)
        self.identity = digest(
            dict(
                rows=[(r['index'], r['row_id'], r['pass_number']) for r in metadata],
                batches=self.batches,
            )
        )

    def __iter__(self):
        yield from (list(batch) for batch in self.batches)

    def __len__(self):
        return len(self.batches)

    def expected(self, step):
        if not 1 <= step <= len(self):
            raise ValueError('Step outside the single-epoch schedule')
        return self.batches[step - 1]

    def verify_batch(self, step, metadata):
        indices = self.expected(step)
        if [(r['index'], r['row_id']) for r in metadata] != [
            (i, self.metadata[i]['row_id']) for i in indices
        ]:
            raise RuntimeError('Dataloader skipped, repeated, or shuffled curriculum rows')


def validate_order_config(config):
    """Reject a composed verl config that alters the curriculum or is not synchronous V1 GRPO."""
    data, trainer = config.data, config.trainer
    if (
        trainer.total_epochs != 1
        or trainer.total_training_steps is not None
        or not trainer.use_v1
        or trainer.v1.trainer_mode != 'sync'
    ):
        raise ValueError('Use synchronous V1, exactly one epoch, and no step-count override')
    if (
        data.shuffle
        or data.filter_overlong_prompts
        or data.train_max_samples != -1
        or data.truncation != 'error'
        or data.dataloader_num_workers != 0
    ):
        raise ValueError(
            'Shuffling, filtering, truncation, workers, or row limits would break the curriculum'
        )
    if config.algorithm.adv_estimator != 'grpo' or config.critic.enable:
        raise ValueError('This migration supports GRPO without a critic')
    if config.actor_rollout_ref.model.use_remove_padding:
        raise ValueError(
            'Packed sequences are not validated for the Qwen3.5 fallback kernels; keep separate padded sequences'
        )
    group = config.actor_rollout_ref.rollout.n
    world = trainer.n_gpus_per_node * trainer.nnodes
    if group < 2 or group % world:
        raise ValueError(
            'GRPO group size must be a multiple of trainer GPU count, including short batches'
        )
    if trainer.save_freq < 1 or config.actor_rollout_ref.actor.checkpoint.get('async_save', False):
        raise ValueError('Require synchronous checkpoints and a positive save frequency')
    if config.actor_rollout_ref.actor.ppo_epochs != 1:
        raise ValueError('Use one actor optimization pass per rollout batch')
    if config.actor_rollout_ref.actor.ppo_mini_batch_size != data.train_batch_size:
        raise ValueError('Use one full rollout batch per optimizer minibatch')
