"""Test the curated i.i.d. run: plan, layout, resume and its consumers, with a fake T2I model."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

from baselines.assets import seed_for
from elicitation.selection import prepare as assign, verified_images
from hpdv2.storage import atomic_json, prompt_id, read_json
from iid.curated import aggregate, prepare, worker

HAS_PIL = importlib.util.find_spec('PIL') is not None


class FakeT2IModel:
    calls = []

    def __init__(self, assets, cfg, run):
        self.cfg = cfg

    def render(self, prompt, seed):
        from PIL import Image

        FakeT2IModel.calls.append((prompt, seed))
        size = self.cfg['resolution']
        return Image.new('RGB', (size, size), (seed % 256, 32, 64)), {'fake': True}

    def close(self):
        pass


@unittest.skipUnless(HAS_PIL, 'Pillow required for image verification tests')
class CuratedTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        prompts = ['a clock', 'a chair', 'a red barn', 'two cats', 'a lighthouse at dusk']
        rows = [
            dict(id=prompt_id(p), prompt=p, cluster_id=str(i), split='train', variation_axes=[])
            for i, p in enumerate(prompts)
        ]
        self.pool = self.root / 'pool.json'
        atomic_json(
            self.pool,
            dict(
                schema_version=2,
                kind='clustered_pool',
                metadata=dict(upstream_complete=True),
                kept=rows,
            ),
        )
        self.run = self.root / 'run'
        FakeT2IModel.calls = []

    def render(self, model, shard, budget=None):
        return worker(self.run, model, shard, budget, factory=FakeT2IModel)

    def test_plan_is_read_only_until_execute_and_writes_every_manifest(self):
        plan = prepare(self.pool, self.run)
        self.assertEqual((plan['prompts'], plan['images_per_t2i_model']), (5, 20))
        self.assertFalse(self.run.exists())
        prepare(self.pool, self.run, execute=True)
        self.assertEqual(len(list(self.run.glob('*/shard-*/manifest.json'))), 32)
        self.assertNotIn('variation_axes', read_json(self.run / 'prompts.json')[0])
        with self.assertRaises(ValueError):
            prepare(self.pool, self.run, execute=True)

    def test_resume_generates_only_missing_images_with_fixed_seeds(self):
        prepare(self.pool, self.run, execute=True)
        rows = sorted(read_json(self.run / 'prompts.json'), key=lambda r: r['id'])
        first = self.render('flux2-klein-9b', 0, budget=3)
        self.assertEqual((first['n_images'], first['complete']), (3, False))
        done = self.render('flux2-klein-9b', 0)
        self.assertTrue(done['complete'])
        self.assertEqual(len(FakeT2IModel.calls), 4)
        self.assertEqual(
            [seed for _, seed in FakeT2IModel.calls],
            [seed_for(rows[0]['id'], 42, i) for i in range(4)],
        )
        slot = self.run / 'flux2-klein-9b/shard-00/images' / rows[0]['id'] / '0002'
        self.assertEqual({p.name for p in slot.iterdir()}, {'image.png', 'receipt.json'})
        self.render('flux2-klein-9b', 0)
        self.assertEqual(len(FakeT2IModel.calls), 4)

    def test_corrupt_image_stops_the_shard(self):
        prepare(self.pool, self.run, execute=True)
        self.render('sd35-large', 1)
        image = next((self.run / 'sd35-large/shard-01/images').glob('*/0000/image.png'))
        image.write_bytes(b'not a png')
        with self.assertRaises(Exception):
            self.render('sd35-large', 1)

    def test_aggregate_and_elicitation_read_the_layout(self):
        prepare(self.pool, self.run, execute=True)
        with self.assertRaises(ValueError):
            aggregate(self.run, 'qwen-image-2512')
        for shard in range(8):
            self.render('qwen-image-2512', shard)
        self.assertEqual(aggregate(self.run, 'qwen-image-2512')['n_images'], 20)
        out = self.root / 'assignment'
        assign(self.run, out, per_t2i_model=1)
        assignment = read_json(out / 'assignment.json')
        row = next(r for r in assignment['items'] if r['renderer'] == 'qwen-image-2512')
        images = verified_images(assignment, row)
        self.assertEqual([image['receipt']['index'] for image in images], [0, 1, 2, 3])
        pending = next(r for r in assignment['items'] if r['renderer'] == 'sd35-medium')
        self.assertIsNone(verified_images(assignment, pending))


if __name__ == '__main__':
    unittest.main()
