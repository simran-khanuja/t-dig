# Portions adapted from verl: Copyright 2024 Bytedance Ltd. and/or its affiliates.
# Licensed under Apache-2.0; see third_party/verl-LICENSE and THIRD_PARTY.md.
"""Adapter around verl's synchronous V1 GRPO trainer, pinned by runtime.lock.json.

The TransferQueue submission and TaskRunner lifecycle follow verl's trainer_base.py and
main_ppo.py (Apache-2.0); verl performs all model training.
"""
from pathlib import Path
import uuid
import numpy as np
import ray
from omegaconf import OmegaConf, open_dict
from torchdata.stateful_dataloader import StatefulDataLoader
import transfer_queue as tq
from verl.trainer.ppo.v1 import AgentLoopManagerTQ
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils import tensordict_utils as tu
from hpdv2.storage import atomic_json, read_json
from .order import PhaseBatchSampler, validate_order_config


class SmokeStop(Exception):
    pass


def reward_extras(field):
    """Return the per-sample `extra_fields` dicts of a TransferQueue fetch as a plain list.

    tensordict 0.10 returns a `NonTensorStack`, which has `tolist()`, when the fetch also
    selects tensor fields, and a `LinkedList`, a `list` subclass without `tolist()`, when
    `extra_fields` is the only field selected. Elements can be wrapped in `NonTensorData`.
    """
    items = field.tolist() if hasattr(field, 'tolist') else list(field)
    return [getattr(item, 'data', item) for item in items]


class StrictReplayBuffer(ReplayBuffer):
    def _sync_metadata_from_transfer_queue(self):
        super()._sync_metadata_from_transfer_queue()
        # verl can sample a partially failed GRPO group, which would shrink the group and shift
        # its mean. Any failure stops the run instead.
        if any(self.failure_keys.values()):
            raise RuntimeError(
                'A rollout/reward failed; stop and resume from the last complete checkpoint'
            )


class OrderedGRPOTrainer(PPOTrainerSync):
    def _build_replay_buffer(self):
        original = super()._build_replay_buffer()
        return StrictReplayBuffer(
            original.trainer_mode,
            original.trainer_config,
            original.max_off_policy_threshold,
            original.max_off_policy_strategy,
            original.sampler_kwargs,
            original.poll_interval,
        )

    def _init_dataloader(self):
        validate_order_config(self.config)
        super()._init_dataloader()
        # Read raw metadata without tokenizing the complete dataset on the driver.
        metadata = list(self.train_dataset.dataframe['extra_info'])
        self.phase_sampler = PhaseBatchSampler(metadata, self.config.data.train_batch_size)
        self.train_dataloader = StatefulDataLoader(
            self.train_dataset,
            batch_sampler=self.phase_sampler,
            num_workers=0,
            collate_fn=collate_fn,
        )
        self.train_dataloader_it = None
        # A step cap trains the first max_steps batches of the full-epoch curriculum. verl ends
        # the loop, and saves a checkpoint, at total_training_steps.
        cap = self.config.tdig.get('max_steps', 0)
        self.total_training_steps = (
            min(len(self.phase_sampler), cap) if cap else len(self.phase_sampler)
        )
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = (
                self.total_training_steps
            )
        atomic_json(
            Path(self.config.trainer.default_local_dir) / 'curriculum.json',
            dict(
                identity=self.phase_sampler.identity,
                rows=len(metadata),
                steps=self.total_training_steps,
                epoch_steps=len(self.phase_sampler),
                batches=[
                    dict(
                        first=b[0],
                        last=b[-1],
                        rows=len(b),
                        pass_number=metadata[b[0]]['pass_number'],
                    )
                    for b in self.phase_sampler.batches
                ],
            ),
        )

    def _add_batch_to_generate(self):
        if self.train_dataloader_it is None:
            self.train_dataloader_it = iter(self.train_dataloader)
        # verl restarts an exhausted dataloader; this override never does within the epoch.
        batch_dict = next(self.train_dataloader_it)
        self.phase_sampler.verify_batch(self.global_steps, list(batch_dict['extra_info']))
        batch_dict['uid'] = np.array(
            [str(uuid.uuid4()) for _ in batch_dict['raw_prompt']], dtype=object
        )
        batch = tu.get_tensordict(batch_dict)
        tu.assign_non_tensor_data(batch, 'global_steps', self.global_steps)
        tags = [dict(is_prompt=True, status='pending', global_steps=self.global_steps)] * len(batch)
        tq.kv_batch_put(keys=list(batch['uid']), partition_id='train', tags=tags)
        self.agent_loop_manager.generate_sequences(batch)

    def step(self, metrics, timing_raw):
        size = len(self.phase_sampler.expected(self.global_steps))
        original = self.config.data.train_batch_size
        # V1 reads the replay sample size and optimizer minibatch size every step, so setting both
        # to this batch's size admits short pass tails without padding or dropped rows.
        self.config.data.train_batch_size = size
        self.config.actor_rollout_ref.actor.ppo_mini_batch_size = size
        try:
            batch = super().step(metrics, timing_raw)
        finally:
            self.config.data.train_batch_size = original
            self.config.actor_rollout_ref.actor.ppo_mini_batch_size = original
        indices = self.phase_sampler.expected(self.global_steps)
        atomic_json(
            Path(self.config.trainer.default_local_dir) / 'steps' / f'{self.global_steps:06d}.json',
            dict(
                step=self.global_steps,
                indices=indices,
                curriculum=self.phase_sampler.identity,
                metrics={
                    k: float(v)
                    for k, v in metrics.items()
                    if isinstance(v, (int, float, np.number))
                },
            ),
        )
        return batch

    def _log_rollout_data(self, batch, timing_raw, rollout_data_dir):
        """Dump the step's rollouts, then write the reward metrics of every completion.

        verl dumps the text and scalar score only, and logs the reward function's metrics
        (diversity, fidelity, tracked scores) for validation batches alone. They are read here
        from the TransferQueue field that validation uses.

        Writes `<checkpoints>/reward-metrics/<step>.json` with fields `step`, `completions`,
        `malformed` (count), `mean_valid` (mean of each metric over completions that parsed)
        and `per_completion` (uid plus metrics, in TransferQueue key order). A malformed
        completion carries zeros for every metric, so it is excluded from `mean_valid`.
        """
        super()._log_rollout_data(batch, timing_raw, rollout_data_dir)
        fields = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=['extra_fields']
        )
        rows = []
        for key, extra in zip(batch.keys, reward_extras(fields['extra_fields']), strict=True):
            info = extra.get('reward_extra_info', {}) if isinstance(extra, dict) else {}
            rows.append(
                dict(
                    uid=key,
                    **{
                        k: float(v)
                        for k, v in info.items()
                        if isinstance(v, (int, float, np.number))
                    },
                )
            )
        valid = [r for r in rows if r.get('malformed', 0.0) == 0.0]
        names = sorted({k for r in valid for k in r if k != 'uid'})
        atomic_json(
            Path(self.config.trainer.default_local_dir)
            / 'reward-metrics'
            / f'{self.global_steps:06d}.json',
            dict(
                step=self.global_steps,
                completions=len(rows),
                malformed=len(rows) - len(valid),
                mean_valid={k: float(np.mean([r[k] for r in valid if k in r])) for k in names},
                per_completion=rows,
            ),
        )

    def _load_checkpoint(self):
        cfg = self.config.trainer
        root = Path(cfg.default_local_dir)
        checkpoint = None
        if cfg.resume_mode == 'resume_path':
            checkpoint = Path(cfg.resume_from_path)
        elif cfg.resume_mode == 'auto' and (root / 'latest_checkpointed_iteration.txt').exists():
            checkpoint = (
                root
                / f"global_step_{int((root/'latest_checkpointed_iteration.txt').read_text().strip())}"
            )
        if checkpoint is not None:
            if (
                not (checkpoint / 'data.pt').is_file()
                or not (checkpoint / 'tdig-complete.json').is_file()
            ):
                raise ValueError(
                    'Incomplete checkpoint: require model, dataloader, and tdig-complete.json completion marker'
                )
            marker = read_json(checkpoint / 'tdig-complete.json')
            if marker['curriculum'] != self.phase_sampler.identity:
                raise ValueError(
                    'Checkpoint curriculum differs from the current dataset/batch size'
                )
            if marker['step'] != int(checkpoint.name.removeprefix('global_step_')):
                raise ValueError('Checkpoint step identity mismatch')
        super()._load_checkpoint()

    def _save_checkpoint(self):
        super()._save_checkpoint()
        atomic_json(
            Path(self.config.trainer.default_local_dir)
            / f'global_step_{self.global_steps}'
            / 'tdig-complete.json',
            dict(curriculum=self.phase_sampler.identity, step=self.global_steps),
        )

    def on_step_end(self):
        super().on_step_end()
        # Extra early checkpoints: every `tdig.save_freq_early` updates up to update
        # `tdig.save_freq_early_until`, in addition to verl's `trainer.save_freq`. The diversity
        # strength scale draws on adapters from these early updates.
        early = self.config.tdig.get('save_freq_early', 0)
        until = self.config.tdig.get('save_freq_early_until', 0)
        if (
            early
            and self.global_steps <= until
            and self.global_steps % early == 0
            and self.global_steps % self.config.trainer.save_freq
        ):
            self._save_checkpoint()
        stop = self.config.tdig.stop_after_step
        if stop and self.global_steps >= stop:
            if self.global_steps % self.config.trainer.save_freq:
                self._save_checkpoint()
            self._shutdown_dump_executor()
            raise SmokeStop(f'Bounded test stopped after checkpoint {self.global_steps}')


@ray.remote
class TaskRunner:
    def run(self, config):
        config.transfer_queue.enable = True
        OmegaConf.resolve(config)
        tq.init(config.transfer_queue)
        try:
            trainer = OrderedGRPOTrainer(config=config)
            trainer.init()
            manager = AgentLoopManagerTQ.create(
                config=config,
                llm_client=trainer.get_llm_client(),
                teacher_client=trainer.get_teacher_client(),
                reward_loop_worker_handles=trainer.get_reward_handles(),
            )
            try:
                trainer.fit(manager)
            except SmokeStop as error:
                print(str(error), flush=True)
        finally:
            tq.close()
