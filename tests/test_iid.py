"""Test PNG verification and the render-service contract, without models, GPUs or network."""

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hpdv2.storage import atomic_json, prompt_id, read_json
from iid.cli import main, parser
from iid.runner import generate, seeds_for
from iid.service import RenderService, INDEPENDENT_STRIDE, SEED_MODULUS

HAS_PIL = importlib.util.find_spec('PIL') is not None


class StubT2IModel:
    def __init__(self):
        self.calls = []
        self.ready = True
        self.fake = False
        self.mutation = {}

    def health(self):
        return {'ready': self.ready, 'fake': self.fake, 'models': ['test-model']}

    def render(self, payload):
        from PIL import Image

        self.calls.append(copy.deepcopy(payload))
        path = Path(payload['out_dir']) / 'image_0.png'
        Image.new(
            'RGB', (payload['resolution'], payload['resolution']), (payload['seed'] % 256, 32, 64)
        ).save(path)
        return {
            **payload,
            'images': [str(path)],
            'fake': False,
            'model_path': '/pinned/model',
            **self.mutation,
        }


@unittest.skipUnless(HAS_PIL, 'Pillow required for image verification tests')
class IIDTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'pool.json'
        self.run = self.root / 'run'
        self.rows = [
            {
                'id': prompt_id(p),
                'prompt': p,
                'cluster_id': str(i),
                'split': 'train',
                'variation_axes': [{'dimension': 'should not leak'}],
            }
            for i, p in enumerate(['a clock', 'a chair'])
        ]
        self.data = {
            'schema_version': 2,
            'kind': 'clustered_pool',
            'metadata': {'upstream_complete': True},
            'kept': self.rows,
        }
        atomic_json(self.source, self.data)
        self.cfg = dict(
            k=4,
            render_url='http://localhost:9999',
            model='test-model',
            model_path='/pinned/model',
            renderer_revision='pinned-test',
            resolution=8,
            steps=2,
            guidance=1.0,
            seed=42,
            timeout=10,
        )
        self.client = StubT2IModel()

    def generate(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return generate(self.source, self.run, self.cfg, client=self.client, **kwargs)

    def test_any_positive_k_and_exact_seeded_sets(self):
        for k in [1, 2, 3, 4, 8, 16]:
            with self.subTest(k=k):
                self.run = self.root / f'k{k}'
                self.cfg['k'] = k
                self.client = StubT2IModel()
                result = self.generate(execute=True, max_requests=2 * k)
                self.assertTrue(result['complete'])
                self.assertEqual(len(self.client.calls), 2 * k)
                s = read_json(self.run / 'summary.json')
                for item in s['items']:
                    self.assertEqual(len(item['images']), k)
                    self.assertEqual(len({im['latent_seed'] for im in item['images']}), k)
                    self.assertNotIn('variation_axes', item)
                    self.assertEqual(item['split'], 'train')
                self.assertTrue(
                    all(p['init_diversity'] == 1 and p['n_total'] == 1 for p in self.client.calls)
                )

    def test_partial_budget_resume_does_not_repeat_images(self):
        first = self.generate(execute=True, max_requests=3)
        self.assertFalse(first['complete'])
        self.assertEqual(first['n_images'], 3)
        second = self.generate(execute=True, max_requests=20)
        self.assertTrue(second['complete'])
        self.assertEqual(len(self.client.calls), 8)
        before = len(self.client.calls)
        self.generate(execute=True, max_requests=10)
        self.assertEqual(len(self.client.calls), before)

    def test_dry_run_and_zero_limit_make_no_service_calls(self):
        self.client.health = lambda: self.fail('network request')
        result = self.generate()
        self.assertEqual(result['n_images_total'], 8)
        self.assertFalse(self.run.exists())
        result = self.generate(execute=True, max_requests=1, limit=0)
        self.assertEqual(result['n_images'], 0)

    def test_configuration_mismatch_before_service_call(self):
        self.generate(execute=True, max_requests=1)
        for key, value in [
            ('k', 8),
            ('steps', 4),
            ('renderer_revision', 'changed'),
            ('seed', 7),
            ('model_path', '/changed'),
        ]:
            old = self.cfg[key]
            self.cfg[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Manifest mismatch'):
                self.generate(execute=True, max_requests=1)
            self.cfg[key] = old
        self.assertEqual(len(self.client.calls), 1)

    def test_corrupt_existing_image_never_silently_regenerated(self):
        self.generate(execute=True, max_requests=1)
        path = next((self.run / 'images').glob('*/*.png'))
        path.write_bytes(b'broken')
        with self.assertRaises(Exception):
            self.generate(execute=True, max_requests=1)
        self.assertEqual(len(self.client.calls), 1)

    def test_valid_but_replaced_image_rejected(self):
        from PIL import Image

        self.generate(execute=True, max_requests=1)
        path = next((self.run / 'images').glob('*/*.png'))
        Image.new('RGB', (8, 8), 'red').save(path)
        with self.assertRaisesRegex(ValueError, 'content changed'):
            self.generate(execute=True, max_requests=1)

    def test_wrong_response_and_fake_service_rejected(self):
        for n, mutation in enumerate(
            [{'model_path': '/wrong'}, {'steps': 999}, {'fake': True}, {'images': []}]
        ):
            self.run = self.root / f'bad{n}'
            self.client.mutation = mutation
            with self.assertRaisesRegex(ValueError, 'Render failed'):
                self.generate(execute=True, max_requests=1)
            self.assertEqual(read_json(self.run / 'summary.json')['metadata']['n_images'], 0)
        self.run = self.root / 'fake'
        self.client.fake = True
        with self.assertRaisesRegex(ValueError, 'T2I model must'):
            self.generate(execute=True, max_requests=1)

    def test_wrong_dimensions_rejected_and_failed_image_retryable(self):
        from PIL import Image

        render = self.client.render

        def wrong_size(payload):
            result = render(payload)
            Image.new('RGB', (4, 4)).save(result['images'][0])
            return result

        self.client.render = wrong_size
        with self.assertRaisesRegex(ValueError, 'Render failed'):
            self.generate(execute=True, max_requests=1)
        self.assertEqual(read_json(self.run / 'summary.json')['metadata']['n_images'], 0)
        self.client.render = render
        self.generate(execute=True, max_requests=1)
        self.assertEqual(read_json(self.run / 'summary.json')['metadata']['n_images'], 1)
        self.assertEqual(self.client.calls[0]['seed'], self.client.calls[1]['seed'])

    def test_missing_receipt_requires_inspection(self):
        self.generate(execute=True, max_requests=1)
        next((self.run / 'images').glob('*/*.json')).unlink()
        with self.assertRaisesRegex(ValueError, 'untracked'):
            self.generate(execute=True, max_requests=1)

    def test_input_contract_and_partial_lineage(self):
        self.data['metadata']['upstream_complete'] = False
        atomic_json(self.source, self.data)
        with self.assertRaisesRegex(ValueError, 'Incomplete curation'):
            self.generate()
        self.generate(execute=True, max_requests=8, allow_partial=True)
        s = read_json(self.run / 'summary.json')
        self.assertTrue(s['metadata']['complete'])
        self.assertFalse(s['metadata']['full_pipeline_complete'])

    def test_seed_stability_and_service_offset(self):
        a = seeds_for(self.rows[0]['id'], 42, 0)
        b = seeds_for(self.rows[0]['id'], 42, 1)
        self.assertNotEqual(a, b)
        self.assertEqual((a[0] + INDEPENDENT_STRIDE) % SEED_MODULUS, a[1])
        self.assertEqual(a, seeds_for(self.rows[0]['id'], 42, 0))

    def test_cli_default_k_and_reject_nonpositive(self):
        argv = [
            '--input',
            str(self.source),
            '--run-dir',
            str(self.run),
            '--t2i-url',
            'http://localhost:1',
            '--model',
            'test-model',
            '--model-path',
            '/pinned/model',
            '--t2i-model-revision',
            'test',
            '--resolution',
            '8',
            '--steps',
            '2',
            '--guidance',
            '1',
        ]
        self.assertEqual(parser().parse_args(argv).k, 4)
        for k in ['0', '-1']:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser().parse_args(argv + ['--k', k])
        with contextlib.redirect_stdout(io.StringIO()), patch.object(
            RenderService, 'request', side_effect=AssertionError
        ):
            self.assertEqual(main(argv), 0)

    def test_http_payload_contract(self):
        response = io.BytesIO(json.dumps({'ready': True}).encode())
        with patch('urllib.request.urlopen', return_value=response) as req:
            self.assertEqual(RenderService('http://localhost:1').health(), {'ready': True})
            self.assertEqual(req.call_args.args[0].full_url, 'http://localhost:1/health')


if __name__ == '__main__':
    unittest.main()
