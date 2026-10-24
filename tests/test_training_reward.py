import asyncio
import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
from unittest.mock import patch

import numpy as np
from hpdv2.storage import digest
from axis_grpo.training.metrics import diversity, score
from axis_grpo.training.protocol import (
    METRICS,
    QUALITY_METRICS,
    length_excess,
    malformed_metrics,
    malformed_score,
    parse_prompts,
    parse_spec,
    seed_for,
    validate_item,
    validate_metrics,
    validate_reward,
)
from axis_grpo.training.reward_fn import compute_score, endpoint_for
from axis_grpo.training.service import Batcher, RewardEngine, make_server
from axis_grpo.training.order import PhaseBatchSampler
from axis_grpo.training.config import settings, validate_service

REWARD = dict(
    metric='pairwise', mode='hinge', fidelity_weight=0.3, fidelity_threshold=0.5, floor='none'
)
SPEC = dict(base_prompt='A red chair', axis='Chair material', tpips_factor='chair material', k=4)


def item(i=0):
    return dict(
        **SPEC,
        prompts=[
            'A red wooden chair',
            'A red metal chair',
            'A red plastic chair',
            'A red velvet chair',
        ],
        seed=42,
        split='train',
        row_id=f'row-{i}',
    )


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.factors = []

    def render(self, items):
        self.calls.append(items)
        return [list(range(it['k'])) for it in items]

    def alignment(self, groups, items):
        return [[0.2, 0.4, 0.6, 0.8] for _ in items]

    def embed(self, images, factor):
        self.factors.append(factor)
        return np.eye(len(images))


class RewardMathTests(unittest.TestCase):
    def test_known_diversities(self):
        d = diversity(np.ones((4, 3)))
        self.assertAlmostEqual(d['pairwise'], 0)
        self.assertAlmostEqual(d['vendi'], 1)
        d = diversity(np.eye(4))
        self.assertAlmostEqual(d['pairwise'], 1)
        self.assertAlmostEqual(d['normalized_vendi'], 1)
        self.assertAlmostEqual(diversity([[1, 0], [-1, 0]])['pairwise'], 2)

    def test_hinge_and_linear(self):
        s = score(np.eye(4), [0.2, 0.4, 0.6, 0.8], 'material', REWARD)
        self.assertAlmostEqual(s['score'], 1 - 0.3 * 0.1)
        self.assertEqual(set(s), set(METRICS))
        self.assertAlmostEqual(
            score(np.eye(4), [0.2, 0.4, 0.6, 0.8], 'material', dict(REWARD, mode='linear'))[
                'score'
            ],
            1.15,
        )

    def test_length_term(self):
        """Hinge on words over the cap, capped per prompt; absent keys change nothing."""
        words = lambda n: ' '.join(['w'] * n)
        prompts = [
            words(60),
            words(120),
            words(66),
            words(300),
        ]  # excess 0, 1, 0.1, 4 -> capped at 3
        self.assertAlmostEqual(length_excess(prompts, 60), (0 + 1 + 0.1 + 3) / 4)
        plain = score(np.eye(4), [0.9, 0.9, 0.9, 0.9], 'material', REWARD, prompts=prompts)
        self.assertEqual((plain['length_penalty'], plain['words_per_prompt']), (0.0, 136.5))
        config = dict(REWARD, length_weight=0.1, length_cap=60)
        long = score(np.eye(4), [0.9, 0.9, 0.9, 0.9], 'material', config, prompts=prompts)
        self.assertAlmostEqual(long['length_penalty'], 0.1025)
        self.assertAlmostEqual(long['score'], plain['score'] - 0.1025)
        # At or under the cap the term is exactly zero, so it never pushes below the cap.
        self.assertEqual(
            score(np.eye(4), [0.9] * 4, 'material', config, prompts=[words(60)] * 4)[
                'length_penalty'
            ],
            0.0,
        )
        # The worst valid reward stays above the malformed penalty.
        worst = score(
            np.ones((4, 3)),
            [0.0] * 4,
            'material',
            dict(config, floor='none'),
            prompts=[words(900)] * 4,
        )['score']
        self.assertGreater(worst, malformed_score(config))
        self.assertEqual(malformed_score(config), -3 - 0.3 - 0.3)
        self.assertEqual(malformed_score(REWARD), -3.3)
        with self.assertRaises(ValueError):
            score(np.eye(4), [0.9] * 4, 'material', config)  # prompts required
        for bad in (
            dict(REWARD, length_weight=0.1),
            dict(REWARD, length_cap=60),
            dict(config, length_weight=-1),
            dict(config, length_cap=0),
            dict(config, length_cap=60.5),
        ):
            with self.assertRaises(ValueError):
                validate_reward(bad)
        self.assertNotEqual(digest(config), digest(REWARD))

    def test_edit_task_drift_term(self):
        """The drift hinge is per image, capped at 1, and needs its two reward keys."""
        config = dict(REWARD, drift_weight=1.0, drift_threshold=0.35)
        plain = score(np.eye(4), [0.9] * 4, 'material', REWARD)
        self.assertEqual((plain['drift_mean'], plain['drift_penalty']), (0.0, 0.0))
        # excess over 0.35: 0, 0.05, 0.25, and 2.0 -> capped at 1; mean = 0.325
        edited = score(np.eye(4), [0.9] * 4, 'material', config, drift=[0.10, 0.40, 0.60, 2.35])
        self.assertAlmostEqual(edited['drift_penalty'], 0.325)
        self.assertAlmostEqual(edited['drift_mean'], 0.8625)
        self.assertAlmostEqual(edited['score'], plain['score'] - 0.325)
        self.assertEqual(set(edited), set(METRICS))
        # An edit service also scores generated sets, which carry no drift.
        self.assertEqual(score(np.eye(4), [0.9] * 4, 'material', config)['drift_penalty'], 0.0)
        with self.assertRaises(ValueError):
            score(np.eye(4), [0.9] * 4, 'material', REWARD, drift=[0.1] * 4)  # no drift keys
        with self.assertRaises(ValueError):
            score(np.eye(4), [0.9] * 4, 'material', config, drift=[0.1] * 3)  # one per image
        self.assertEqual(malformed_score(config), -3 - 0.3 - 1.0)
        worst = score(np.ones((4, 3)), [0.0] * 4, 'material', config, drift=[5.0] * 4)['score']
        self.assertGreater(worst, malformed_score(config))
        for bad in (
            dict(REWARD, drift_weight=1.0),
            dict(config, drift_threshold=-0.1),
            dict(config, drift_weight='1'),
        ):
            with self.assertRaises(ValueError):
                validate_reward(bad)

    def test_row_threshold_replaces_the_configured_one(self):
        s = score(np.eye(4), [0.2, 0.4, 0.6, 0.8], 'material', REWARD)
        self.assertEqual(s['fidelity_threshold'], 0.5)
        s = score(np.eye(4), [0.2, 0.4, 0.6, 0.8], 'material', REWARD, threshold=0.9)
        self.assertAlmostEqual(s['fidelity_penalty'], 0.4)
        self.assertAlmostEqual(s['score'], 1 - 0.3 * 0.4)
        self.assertEqual(s['fidelity_threshold'], 0.9)
        with self.assertRaises(ValueError):
            score(np.eye(4), [0.2, 0.4, 0.6, 0.8], 'material', REWARD, threshold=1.5)

    def test_overall_does_not_cancel_its_own_reward(self):
        for metric in ['pairwise', 'vendi']:
            cfg = dict(REWARD, floor='overall', metric=metric)
            self.assertEqual(score(np.eye(4), [1] * 4, 'overall', cfg)['floor'], 0)
            self.assertAlmostEqual(score(np.eye(4), [1] * 4, 'overall', cfg)['score'], 1)
            self.assertAlmostEqual(
                score(np.eye(4), [1] * 4, 'chair material', cfg, np.eye(4))['score'], 0
            )

    def test_vendi_floor_uses_vendi(self):
        cfg = dict(REWARD, floor='overall', metric='vendi')
        overall = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        s = score(np.eye(4), [1] * 4, 'material', cfg, overall)
        self.assertAlmostEqual(s['floor'], 1 / 3)
        self.assertAlmostEqual(s['score'], 2 / 3)

    def test_invalid_scorer_outputs_raise(self):
        for vectors in [np.zeros((4, 3)), np.full((4, 3), np.nan), np.ones((1, 3))]:
            with self.assertRaises(ValueError):
                diversity(vectors)
        for alignment in [[1, 1], [-0.1] * 4, [float('nan')] * 4, [1.1] * 4]:
            with self.assertRaises(ValueError):
                score(np.eye(4), alignment, 'x', REWARD)

    def test_bad_format_lower_than_valid_reward_even_with_floor(self):
        cfg = dict(REWARD, fidelity_weight=10, floor='overall')
        valid = score(np.ones((4, 3)), [0] * 4, 'x', cfg, np.eye(4))['score']
        self.assertGreater(valid, malformed_score(cfg))


class ContractTests(unittest.TestCase):
    def test_tags(self):
        self.assertEqual(parse_prompts('<prompt>a</prompt>\n<prompt>b</prompt>', 2), ['a', 'b'])
        for s in [
            '<prompt>a</prompt>',
            'hello <prompt>a</prompt><prompt>b</prompt>',
            '<prompt></prompt><prompt>b</prompt>',
            '<prompt><prompt>a</prompt></prompt>',
        ]:
            with self.assertRaises(ValueError):
                parse_prompts(s, 2)

    def test_schema_and_seeds(self):
        self.assertEqual(parse_spec(json.dumps(SPEC)), SPEC)
        self.assertEqual(validate_item(item()), item())
        self.assertEqual(seed_for('x', 'a'), seed_for('x', 'b'))
        self.assertNotEqual(seed_for('x', 'a', scope='pair'), seed_for('x', 'b', scope='pair'))
        with self.assertRaises(ValueError):
            parse_spec(dict(SPEC, axis='overall'))
        with self.assertRaises(ValueError):
            validate_item(dict(item(), seed=True))
        self.assertEqual(parse_spec(dict(SPEC, align_thresh=0.83))['align_thresh'], 0.83)
        self.assertEqual(validate_item(dict(item(), align_thresh=0.83))['align_thresh'], 0.83)
        for bad in (1.5, -0.1, True, '0.8', float('nan')):
            with self.assertRaises(ValueError):
                parse_spec(dict(SPEC, align_thresh=bad))
        with self.assertRaises(ValueError):
            parse_spec(dict(SPEC, guidance=4.5))
        # The editing task: k source images, one per instruction, always with task='edit'.
        edit = dict(SPEC, task='edit', source_images=['/a.png', '/b.png', '/c.png', '/d.png'])
        self.assertEqual(parse_spec(edit), edit)
        self.assertEqual(
            validate_item(dict(item(), task='edit', source_images=edit['source_images']))['task'],
            'edit',
        )
        for bad in (
            dict(SPEC, task='edit'),
            dict(SPEC, source_images=edit['source_images']),
            dict(edit, task='inpaint'),
            dict(edit, source_images=edit['source_images'][:3]),
            dict(edit, source_images=['', 1, 2, 3]),
        ):
            with self.assertRaises(ValueError):
                parse_spec(bad)

    def test_routing(self):
        routes = {'qwen': ['http://one', 'http://two']}
        self.assertEqual(
            [endpoint_for(routes, 'qwen') for _ in range(3)],
            ['http://one', 'http://two', 'http://one'],
        )
        with self.assertRaises(ValueError):
            endpoint_for(routes, 'missing')

    def test_t2i_model_setting_typos_cannot_silently_use_defaults(self):
        r = dict(
            resolution=512, steps=16, guidance=4.5, max_sequence_length=256, negative_prompt=''
        )
        cfg = dict(
            renderer='sd35-medium',
            renderer_assets='/fixture',
            render=dict(train=r, val=r),
            device='cuda:0',
            tpips_checkpoint='/fixture',
            cache='/fixture',
            tpips_batch=4,
            render_batch=8,
            staging='/fixture',
            vqa_url='http://localhost:1234',
            vqa_model='clip-flant5-xxl',
            timeout=1800,
            reward=REWARD,
            batch_items=4,
            batch_wait_ms=20,
            save_images=False,
        )
        validate_service(cfg)
        bad = copy.deepcopy(cfg)
        bad['render']['train']['step'] = bad['render']['train'].pop('steps')
        with self.assertRaises(ValueError):
            validate_service(bad)
        with self.assertRaises(ValueError):
            validate_service(dict(cfg, renderer='flux2-klein-9b'))
        # Tracked scorers are optional; the aesthetic scorer needs its head file.
        validate_service(dict(cfg, track_metrics=['imagereward', 'hpsv2', 'clip_score']))
        with self.assertRaises(ValueError):
            validate_service(dict(cfg, track_metrics=['pickscore']))
        with self.assertRaises(ValueError):
            validate_service(dict(cfg, track_metrics=['aesthetic'], aesthetic_head='/missing.pth'))


class QualityTermTests(unittest.TestCase):
    def test_quality_hinge_charges_only_the_deficit_past_the_margin_and_keeps_old_schema(self):
        vectors = np.eye(4)
        alignment = [0.9] * 4
        plain = score(vectors, alignment, 'overall', REWARD)
        self.assertFalse(set(QUALITY_METRICS) & set(plain))
        cfg = dict(REWARD, quality_weight=0.05, quality_margin=1.0)
        free = score(
            vectors, alignment, 'overall', cfg, quality=[11.5, 12, 13, 11], quality_threshold=12.0
        )
        self.assertEqual(free['quality_penalty'], 0.0)
        self.assertAlmostEqual(free['reward'], plain['reward'])
        hit = score(
            vectors, alignment, 'overall', cfg, quality=[9, 9, 9, -50], quality_threshold=12.0
        )
        self.assertAlmostEqual(hit['quality_penalty'], 0.05 * (2 + 2 + 2 + 6) / 4)
        self.assertAlmostEqual(hit['reward'], plain['reward'] - hit['quality_penalty'])
        with self.assertRaises(ValueError):
            score(vectors, alignment, 'overall', cfg)
        self.assertLess(malformed_score(cfg), malformed_score(REWARD))
        self.assertEqual(set(malformed_metrics(cfg)), set(hit))
        with self.assertRaises(ValueError):
            score(vectors, alignment, 'overall', dict(REWARD, quality_weight=0.05))


class RowLengthCapTests(unittest.TestCase):
    def test_a_row_cap_replaces_the_configured_cap(self):
        cfg = dict(REWARD, length_weight=0.1, length_cap=150)
        prompts = ['w ' * 40] * 4
        self.assertEqual(
            score(np.eye(4), [0.9] * 4, 'overall', cfg, prompts=prompts)['length_penalty'], 0.0
        )
        self.assertAlmostEqual(
            score(np.eye(4), [0.9] * 4, 'overall', cfg, prompts=prompts, length_cap=20)[
                'length_penalty'
            ],
            0.1 * 1.0,
        )
        spec = dict(base_prompt='a dog', axis='overall', tpips_factor='overall', k=4, length_cap=22)
        self.assertEqual(parse_spec(spec)['length_cap'], 22)
        with self.assertRaises(ValueError):
            parse_spec(dict(spec, length_cap=0))


class ScoreCallerTests(unittest.TestCase):
    def test_every_caller_of_the_reward_passes_the_prompts(self):
        """Every call to score() in the training package passes the prompts the length term needs."""
        import ast

        root = Path(__file__).resolve().parents[1] / 'axis_grpo/training'
        calls = []
        for path in sorted(root.glob('*.py')):
            if path.name == 'hpsv3_model.py':
                continue  # its own score() is the HPSv3 scorer, not the reward
            for node in ast.walk(ast.parse(path.read_text())):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == 'score'
                ):
                    supplied = {k.arg for k in node.keywords} | (
                        {'prompts'} if len(node.args) >= 7 else set()
                    )
                    calls.append((path.name, node.lineno, 'prompts' in supplied))
        self.assertGreaterEqual(len(calls), 2, calls)
        self.assertEqual([c for c in calls if not c[2]], [])


class RolloutMetricTests(unittest.TestCase):
    def test_reward_extras_accepts_both_tensordict_containers(self):
        """reward_extras reads both a tensordict index result (no tolist()) and a get() NonTensorStack."""
        import torch
        from tensordict import TensorDict, NonTensorStack, NonTensorData
        from axis_grpo.training.trainer import reward_extras

        infos = [
            {'reward_extra_info': {'score': 0.5, 'malformed': 0.0}},
            {'reward_extra_info': {'score': -4.0, 'malformed': 1.0}},
        ]
        td = TensorDict(
            {
                'extra_fields': NonTensorStack(*[NonTensorData(x) for x in infos]),
                'rm_scores': torch.zeros(2, 3),
            },
            batch_size=[2],
        )
        self.assertFalse(hasattr(td['extra_fields'], 'tolist'))
        self.assertEqual(reward_extras(td['extra_fields']), infos)
        self.assertEqual(reward_extras(td.get('extra_fields')), infos)
        self.assertEqual(reward_extras(list(infos)), infos)


class TrainingSettingsTests(unittest.TestCase):
    def test_offline_validation_disables_inline_passes(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / 'policy').mkdir()
            (d / 'policy/config.json').write_text('{}')
            for name in ('train.parquet', 'val.parquet'):
                (d / name).write_text('')
            base = dict(
                policy_path=str(d / 'policy'),
                train_file=str(d / 'train.parquet'),
                val_file=str(d / 'val.parquet'),
                cache=str(d),
                reward_endpoints=['http://node:1'],
                save_freq=50,
            )
            cfg = settings(
                dict(
                    base,
                    validation='offline',
                    test_freq=0,
                    val_before_train=False,
                    track_metrics=['imagereward', 'hpsv2', 'clip_score', 'aesthetic'],
                )
            )
            self.assertEqual(
                (cfg['save_freq'], cfg['test_freq'], cfg['val_before_train']), (50, 0, False)
            )
            # Offline means no inline pass at all; inline production still requires both.
            with self.assertRaises(ValueError):
                settings(dict(base, validation='offline', test_freq=500, val_before_train=False))
            with self.assertRaises(ValueError):
                settings(dict(base, validation='inline', test_freq=0, val_before_train=False))
            with self.assertRaises(ValueError):
                settings(dict(base, validation='later'))
            with self.assertRaises(ValueError):
                settings(dict(base, track_metrics=['unknown']))
            # A step cap is part of the settings but not of the identity compared on resume,
            # so a finished 1,200-step run can be extended.
            from axis_grpo.training.run import stable_identity

            self.assertEqual(settings(dict(base, max_steps=1200, test_freq=25))['max_steps'], 1200)
            with self.assertRaises(ValueError):
                settings(dict(base, max_steps=-1))
            short, long = (
                dict(config=settings(dict(base, max_steps=n, test_freq=25))) for n in (1200, 2400)
            )
            self.assertEqual(stable_identity(short), stable_identity(long))
            self.assertNotEqual(
                stable_identity(short),
                stable_identity(dict(config=dict(short['config'], learning_rate=1e-5))),
            )


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.backend = FakeBackend()
        self.config = dict(renderer='sd35-medium', reward=REWARD, timeout=3)
        self.engine = RewardEngine(self.backend, self.config, self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_tracked_scores_are_logged_and_never_change_the_reward(self):
        plain = self.engine.score_many([item()])[0]['metrics']
        self.backend.tracked = lambda images, text: dict(
            imagereward=[1.0, 0.0, 1.0, 0.0], clip_score=[0.2, 0.2, 0.4, 0.4]
        )
        config = dict(self.config, track_metrics=['imagereward', 'clip_score'])
        with tempfile.TemporaryDirectory() as d:
            metrics = RewardEngine(self.backend, config, Path(d)).score_many([item()])[0]['metrics']
            saved = json.loads(next((Path(d) / 'results').iterdir()).read_text())
        self.assertEqual(metrics['score'], plain['score'])
        self.assertAlmostEqual(metrics['imagereward'], 0.5)
        self.assertAlmostEqual(metrics['clip_score'], 0.3)
        self.assertEqual(set(metrics), set(METRICS) | {'imagereward', 'clip_score'})
        # Per-image values are kept beside the means.
        self.assertEqual(saved['detail']['tracked']['imagereward'], [1.0, 0.0, 1.0, 0.0])
        self.assertEqual(saved['detail']['alignment'], [0.2, 0.4, 0.6, 0.8])
        # A malformed completion carries the same keys, or the trainer cannot average them.
        self.assertEqual(
            set(malformed_metrics(REWARD, tracked=['imagereward', 'clip_score'])), set(metrics)
        )
        with self.assertRaises(ValueError):
            validate_metrics(metrics)
        with self.assertRaises(ValueError):
            malformed_metrics(REWARD, tracked=['unknown'])
        self.assertIn('hpsv3', malformed_metrics(REWARD, tracked=['hpsv3']))

    def test_service_applies_the_length_term_to_the_expanded_prompts(self):
        config = dict(self.config, reward=dict(REWARD, length_weight=0.1, length_cap=2))
        with tempfile.TemporaryDirectory() as d:
            metrics = RewardEngine(self.backend, config, Path(d)).score_many([item()])[0]['metrics']
        plain = self.engine.score_many([item()])[0]['metrics']
        # Each fixture prompt has four words against a cap of two: excess 1.0, penalty 0.1.
        self.assertAlmostEqual(metrics['length_penalty'], 0.1)
        self.assertEqual(metrics['words_per_prompt'], 4.0)
        self.assertAlmostEqual(metrics['score'], plain['score'] - 0.1)

    def test_service_scores_an_edit_item_with_its_drift(self):
        """An edit item is rendered by the backend's edit path and scored with the drift term."""
        seen = []
        self.backend.drift = lambda it, images: seen.append(it['source_images']) or [
            0.35,
            0.45,
            0.55,
            0.35,
        ]
        config = dict(self.config, reward=dict(REWARD, drift_weight=1.0, drift_threshold=0.35))
        edit = dict(item(), task='edit', source_images=['/s0.png', '/s1.png', '/s2.png', '/s3.png'])
        with tempfile.TemporaryDirectory() as d:
            out = RewardEngine(self.backend, config, Path(d)).score_many([edit, item(1)])
            saved = [json.loads(p.read_text()) for p in (Path(d) / 'results').iterdir()]
        plain = self.engine.score_many([item()])[0]['metrics']['score']
        self.assertAlmostEqual(out[0]['metrics']['drift_penalty'], 0.075)  # mean of 0, .1, .2, 0
        self.assertAlmostEqual(out[0]['metrics']['score'], plain - 0.075)
        self.assertEqual(out[1]['metrics']['drift_penalty'], 0.0)  # the generated set beside it
        self.assertEqual(seen, [edit['source_images']])
        self.assertEqual(sorted(len(r['detail'].get('drift', [])) for r in saved), [0, 4])

    def test_durable_reuse_and_config_drift(self):
        answer = self.engine.score_many([item()])
        restart = RewardEngine(self.backend, self.config, self.root)
        self.assertEqual(restart.score_many([item()]), answer)
        self.assertEqual(len(self.backend.calls), 1)
        with self.assertRaises(ValueError):
            RewardEngine(self.backend, dict(self.config, renderer='qwen'), self.root).score_many(
                [item()]
            )

    def test_batches_deduplicate_inflight(self):
        b = Batcher(self.engine, max_items=4, wait_ms=100)
        try:
            first = b.submit(item())
            duplicate = b.submit(item())
            other = b.submit(item(1))
            self.assertIs(first, duplicate)
            first.result(3)
            other.result(3)
            self.assertEqual(len(self.backend.calls), 1)
            self.assertEqual(len(self.backend.calls[0]), 2)
        finally:
            b.close()

    def test_exception_is_not_cached_as_reward(self):
        with patch.object(self.backend, 'alignment', side_effect=RuntimeError('scorer failed')):
            with self.assertRaisesRegex(RuntimeError, 'scorer failed'):
                self.engine.score_many([item()])
        self.assertFalse((self.root / 'results').exists())

    def test_http_client_roundtrip_and_fault(self):
        b = Batcher(self.engine, wait_ms=1)
        server = make_server('127.0.0.1', 0, b, self.config)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        args = dict(
            data_source='axis_diversity',
            ground_truth=SPEC,
            solution_str=''.join(f'<prompt>{p}</prompt>' for p in item()['prompts']),
            extra_info=dict(row_id='r', source_renderer='qwen-image-2512'),
            routes={'sd35-medium': [f'http://127.0.0.1:{server.server_port}']},
            reward=REWARD,
        )
        try:
            self.assertAlmostEqual(asyncio.run(compute_score(**args))['score'], 0.97)
            # A row threshold travels dataset -> client -> HTTP -> service -> reward math.
            custom = asyncio.run(
                compute_score(**dict(args, ground_truth=dict(SPEC, align_thresh=0.9)))
            )
            self.assertAlmostEqual(custom['score'], 1 - 0.3 * 0.4)
            self.assertEqual(custom['fidelity_threshold'], 0.9)
            with self.assertRaisesRegex(RuntimeError, 'configuration changed'):
                asyncio.run(
                    compute_score(
                        **dict(
                            args, service_hashes={f'http://127.0.0.1:{server.server_port}': 'wrong'}
                        )
                    )
                )
            self.assertEqual(
                asyncio.run(compute_score(**dict(args, solution_str='bad')))['malformed'], 1
            )
            # A provider fault must propagate through the real HTTP client.
            with patch.object(self.backend, 'render', side_effect=RuntimeError('GPU failed')):
                with self.assertRaises(urllib.error.HTTPError):
                    asyncio.run(compute_score(**dict(args, extra_info=dict(row_id='uncached'))))
        finally:
            server.shutdown()
            server.server_close()
            b.close()
            thread.join()


class OrderTests(unittest.TestCase):
    def test_full_curriculum_retains_each_phase_tail(self):
        meta = [dict(index=i, row_id=str(i), pass_number=1 + i // 20500) for i in range(82000)]
        sampler = PhaseBatchSampler(meta, 8)
        batches = list(sampler)
        self.assertEqual(len(batches), 10252)
        self.assertEqual([i for batch in batches for i in batch], list(range(82000)))
        self.assertEqual([len(b) for b in batches if len(b) < 8], [4, 4, 4, 4])
        self.assertTrue(all(len({meta[i]['pass_number'] for i in b}) == 1 for b in batches))

    def test_short_tails_and_phase_boundaries(self):
        meta = [
            dict(index=i, row_id=str(i), pass_number=p)
            for i, p in enumerate([1] * 5 + [2] * 3 + [4] * 2)
        ]
        sampler = PhaseBatchSampler(meta, 4)
        self.assertEqual(list(sampler), [[0, 1, 2, 3], [4], [5, 6, 7], [8, 9]])
        for step, indices in enumerate(sampler, 1):
            sampler.verify_batch(step, [meta[i] for i in indices])
        with self.assertRaises(RuntimeError):
            sampler.verify_batch(2, [meta[0]])
        with self.assertRaises(ValueError):
            PhaseBatchSampler(list(reversed(meta)), 4)
        with self.assertRaises(ValueError):
            sampler.expected(5)

    def test_stateful_resume_keeps_every_row(self):
        try:
            from torchdata.stateful_dataloader import StatefulDataLoader
        except ImportError:
            self.skipTest('Trainer environment required')
        meta = [dict(index=i, row_id=str(i), pass_number=1 + i // 11) for i in range(40)]

        def loader():
            return StatefulDataLoader(
                list(range(40)), batch_sampler=PhaseBatchSampler(meta, 8), num_workers=0
            )

        dl = loader()
        it = iter(dl)
        first = next(it).tolist() + next(it).tolist() + next(it).tolist()
        state = dl.state_dict()
        restored = loader()
        restored.load_state_dict(state)
        rest = [int(i) for batch in restored for i in batch]
        self.assertEqual(first + rest, list(range(40)))


if __name__ == '__main__':
    unittest.main()
