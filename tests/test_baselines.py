import copy
import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

from baselines.config import defaults, validate
from baselines.assets import lock_assets, seed_for
from baselines.algorithms import (
    gamma,
    corrupt,
    clean_prediction,
    next_size,
    solve_subset,
)
from baselines.runner import generate
from baselines.metrics import set_metrics
from benchmarks.coco import select_rows
from hpdv2.storage import read_json, atomic_json


class FakeBackend:
    calls = []
    fail_at = None

    def __init__(self, assets, cfg, run):
        self.cfg = cfg

    def adapt(self, row):
        return [row["prompt"]] * self.cfg["k"]

    def render(self, prompt, seed, noise_seed):
        self.calls.append(seed)
        if self.fail_at == len(self.calls):
            raise RuntimeError("simulated interruption")
        return Image.new("RGB", (self.cfg["resolution"],) * 2, (seed % 255, 20, 50)), {"test": True}

    def group(self, row):
        ids = list(
            range(
                self.cfg["group"]["candidates"] - self.cfg["k"],
                self.cfg["group"]["candidates"],
            )
        )
        return (
            [Image.new("RGB", (self.cfg["resolution"],) * 2, (i, 20, 50)) for i in ids],
            ids,
            {"selected_ids": ids},
        )

    def close(self):
        pass


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        model = self.root / "model"
        model.mkdir()
        (model / "weights").write_text("fixture")
        self.lock = self.root / "assets.json"
        lock_assets(
            {k: model for k in ["renderer", "clip", "dino", "pag", "tokenizer", "pag_runtime"]},
            self.lock,
        )
        self.rows = [
            {
                "id": "coco2017:000000000001",
                "index": 0,
                "prompt": "a bench",
                "metadata": {"prompt": "a bench", "image_id": 1},
            }
        ]
        self.bundle = {"count": 1, "benchmark": "coco2017-val-5000"}
        self.dataset = patch("baselines.runner.dataset", return_value=(self.bundle, self.rows))
        self.dataset.start()
        FakeBackend.calls = []
        FakeBackend.fail_at = None
        self.cfg = defaults("iid")
        self.cfg.update(resolution=32, steps=2, k=3)

    def tearDown(self):
        self.dataset.stop()
        self.temp.cleanup()

    def run_generation(self, cfg=None, **kwargs):
        return generate(
            self.root / "bundle",
            self.root / "run",
            cfg or self.cfg,
            self.lock,
            factory=FakeBackend,
            **kwargs,
        )

    def test_dry_run_and_zero_limit_do_not_load_models(self):
        plan = self.run_generation()
        self.assertEqual(plan["pending_images"], 3)
        self.assertFalse((self.root / "run").exists())
        result = self.run_generation(execute=True, max_new_images=2, limit=0)
        self.assertTrue(result["complete"])
        self.assertEqual(FakeBackend.calls, [])

    def test_partial_resume_preserves_seed_and_prompt_identity(self):
        a = self.run_generation(execute=True, max_new_images=1)
        self.assertEqual(a["verified_images"], 1)
        self.assertFalse((self.root / "run/images/00000").exists())
        b = self.run_generation(execute=True, max_new_images=5)
        self.assertTrue(b["complete"])
        self.assertEqual(len(FakeBackend.calls), 3)
        self.run_generation(execute=True, max_new_images=5)
        self.assertEqual(len(FakeBackend.calls), 3)
        self.assertEqual(
            read_json(self.root / "run/images/00000/metadata.jsonl"),
            self.rows[0]["metadata"],
        )

    def test_mid_generation_failure_does_not_resend_success(self):
        FakeBackend.fail_at = 2
        with self.assertRaises(RuntimeError):
            self.run_generation(execute=True, max_new_images=3)
        first = FakeBackend.calls[0]
        FakeBackend.fail_at = None
        self.run_generation(execute=True, max_new_images=3)
        self.assertEqual(FakeBackend.calls.count(first), 1)

    def test_changed_config_refused(self):
        self.run_generation(execute=True, max_new_images=1)
        altered = copy.deepcopy(self.cfg)
        altered["guidance"] = 5
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            self.run_generation(altered)

    def test_image_corruption_refused(self):
        self.run_generation(execute=True, max_new_images=1)
        Image.new("RGB", (32, 32), "red").save(self.root / "run/work/00000/slots/0000/image.png")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.run_generation()

    def test_changed_adaptation_refused(self):
        self.run_generation(execute=True, max_new_images=1)
        p = self.root / "run/work/00000/adaptations.json"
        a = read_json(p)
        a["prompts"][0] = "a cat"
        atomic_json(p, a)
        with self.assertRaises(ValueError):
            self.run_generation()

    def test_group_atomic_budget_and_original_candidate_seeds(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["method"] = "group"
        cfg["group"]["candidates"] = 7
        a = self.run_generation(cfg, execute=True, max_new_images=2)
        self.assertEqual(a["verified_images"], 0)
        b = self.run_generation(cfg, execute=True, max_new_images=3)
        self.assertTrue(b["complete"])
        for i, candidate in enumerate([4, 5, 6]):
            r = read_json(self.root / f"run/work/00000/slots/{i:04d}/receipt.json")
            self.assertEqual(r["candidate_id"], candidate)
            self.assertEqual(r["seed"], seed_for(self.rows[0]["id"], cfg["seed"], candidate))
        self.run_generation(cfg)

    def test_mixed_group_trace_rejected(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["method"] = "group"
        self.run_generation(cfg, execute=True, max_new_images=3)
        atomic_json(self.root / "run/work/00000/group_trace.json", {"changed": True})
        with self.assertRaisesRegex(ValueError, "trace changed"):
            self.run_generation(cfg)

    def test_asset_content_change_refused(self):
        (self.root / "model/weights").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Asset content changed"):
            self.run_generation(execute=True, max_new_images=3)

    def test_invalid_config(self):
        for key, value in [
            ("k", 0),
            ("steps", True),
            ("resolution", 31),
            ("guidance", float("nan")),
        ]:
            cfg = copy.deepcopy(self.cfg)
            cfg[key] = value
            with self.assertRaises(ValueError):
                validate(cfg)

    def test_seed_independent_of_k_and_method(self):
        a = [seed_for("p", 42, i) for i in range(4)]
        self.assertEqual(a, [seed_for("p", 42, i) for i in range(8)][:4])
        self.assertEqual(len(set(a)), 4)
        self.assertNotEqual(a[0], seed_for("p", 42, 0, "cads-noise"))

    def test_explicit_subset_validation_and_snapshot(self):
        with self.assertRaises(ValueError):
            self.run_generation(indices=[9])
        with self.assertRaises(ValueError):
            self.run_generation(indices=[0, 0])
        with self.assertRaises(ValueError):
            self.run_generation(indices=[0], limit=1)
        self.run_generation(indices=[0], execute=True, max_new_images=3)
        self.assertTrue((self.root / "run/source/baselines/runner.py").is_file())

    def test_corrupt_saved_input_refused(self):
        self.run_generation(execute=True, max_new_images=1)
        rows = copy.deepcopy(self.rows)
        rows[0]["prompt"] = "changed"
        atomic_json(self.root / "run/input.json", rows)
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            self.run_generation()


class MethodMathTests(unittest.TestCase):
    def test_cads_schedule_and_identity(self):
        import torch

        self.assertEqual([gamma(t, 0.6, 0.9) for t in [0, 0.6, 0.9, 1]], [1, 1, 0, 0])
        x = torch.randn(2, 4, 6)
        y = corrupt(x, 1, 0.25, 1, torch.Generator().manual_seed(5))
        self.assertIs(x, y)
        original = x.clone()
        y = corrupt(x, 0.3, 0.25, 1, torch.Generator().manual_seed(5))
        self.assertTrue(torch.equal(x, original))
        self.assertTrue(torch.allclose(x.mean((1, 2)), y.mean((1, 2)), atol=1e-6))
        self.assertTrue(torch.allclose(x.std((1, 2)), y.std((1, 2)), atol=1e-6))
        self.assertFalse(torch.equal(x, y))
        self.assertTrue(torch.equal(y, corrupt(x, 0.3, 0.25, 1, torch.Generator().manual_seed(5))))

    def test_clean_prediction_uses_pre_step_latent(self):
        import torch

        x0 = torch.tensor([2.0])
        noise = torch.tensor([6.0])
        sigma = 0.8
        x = (1 - sigma) * x0 + sigma * noise
        v = noise - x0
        self.assertTrue(torch.allclose(clean_prediction(x, v, sigma), x0))
        self.assertFalse(torch.allclose(clean_prediction(x - 0.1 * v, v, sigma), x0))

    def test_pruning_reaches_exact_k(self):
        for k in [1, 2, 4, 8, 16]:
            n = 128
            for i in range(4):
                n = next_size(n, k, 0.75, last=i == 3)
            self.assertEqual(n, k)

    def test_solver_matches_exhaustive_objective(self):
        rng = np.random.default_rng(7)
        u = rng.normal(size=6)
        d = rng.random((6, 6))
        weight = 0.8
        selected, report = solve_subset(u, d, 3, weight, 10)
        objective = lambda subset: sum(u[i] for i in subset) + weight * sum(
            d[i, j] for i, j in itertools.combinations(subset, 2)
        )
        best = max(objective(v) for v in itertools.combinations(range(6), 3))
        self.assertAlmostEqual(objective(selected), best, places=7)
        self.assertTrue(report["optimal"])

    def test_vendi_identical_and_orthogonal(self):
        for cls, expected in [(np.ones((4, 3)), 1), (np.eye(4), 4)]:
            metrics = set_metrics(
                {
                    "clip": np.ones(4) * 0.3,
                    "pairwise": np.zeros((4, 4)),
                    "dino_cls": cls,
                }
            )
            self.assertAlmostEqual(metrics["dino_cls_vendi"], expected, places=6)
        with self.assertRaises(ValueError):
            set_metrics({"clip": [np.nan], "pairwise": [[0]], "dino_cls": [[1, 0]]})

    def test_coco_selection_order_independent_and_ids_preserved(self):
        images = [{"id": i, "file_name": f"{i:012d}.jpg"} for i in [20, 10]]
        captions = [
            {"id": i * 10 + j, "image_id": i, "caption": f"caption {j} of image {i}"}
            for i in [20, 10]
            for j in range(5)
        ]
        a = select_rows({"images": images, "annotations": captions}, 42, 2)
        b = select_rows({"images": images[::-1], "annotations": captions[::-1]}, 42, 2)
        self.assertEqual(a, b)
        self.assertEqual([r["metadata"]["image_id"] for r in a], [10, 20])
        with self.assertRaises(ValueError):
            select_rows({"images": images, "annotations": captions + captions[:1]}, 42, 2)
