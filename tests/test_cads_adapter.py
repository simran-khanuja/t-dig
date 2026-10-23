"""Test the SD3 CADS adapter against the two SD3-specific causes of CADS collapse."""

import copy
from types import SimpleNamespace
import unittest

import torch

from baselines.algorithms import cads_time, gamma
from baselines.config import defaults, validate
from baselines.sd3 import SD35


class Recorder(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.calls = []
        self.fail = fail

    def forward(self, **kwargs):
        self.calls.append({k: v.clone() for k, v in kwargs.items()})
        if self.fail:
            raise RuntimeError("Injected transformer failure")
        return (torch.zeros(2, 1),)


class FakePipe:
    def __init__(self, fail=False):
        self.transformer = Recorder(fail)
        self.scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
        self.tokens = torch.arange(24).reshape(2, 3, 4).float() / 10
        self.pooled = torch.arange(12).reshape(2, 6).float() / 10

    def __call__(self, **kwargs):
        for time in [1000, 950, 900, 800]:
            self.transformer(
                timestep=torch.tensor([time, time]),
                encoder_hidden_states=self.tokens,
                pooled_projections=self.pooled,
            )
        return SimpleNamespace(images=[object()])


class CadsAdapterTests(unittest.TestCase):
    def model(self, fail=False):
        model = SD35.__new__(SD35)
        model.cfg = defaults("cads")
        model.cfg["steps"] = 4
        model.pipe = FakePipe(fail)
        return model

    def test_tokens_both_branches_pooled_clean_and_no_cumulative_mutation(self):
        model = self.model()
        tokens, pooled = model.pipe.tokens.clone(), model.pipe.pooled.clone()
        _, report = model.render("fixture", 1, 2)
        calls = model.pipe.transformer.calls
        for call in calls:
            self.assertTrue(torch.equal(call["pooled_projections"], pooled))
        for branch in [0, 1]:
            self.assertFalse(torch.equal(calls[0]["encoder_hidden_states"][branch], tokens[branch]))
        self.assertTrue(torch.equal(calls[-1]["encoder_hidden_states"], tokens))
        self.assertTrue(torch.equal(model.pipe.tokens, tokens))
        self.assertTrue(torch.equal(model.pipe.pooled, pooled))
        self.assertEqual(calls[-1]["timestep"].tolist(), [800, 800])
        self.assertEqual(
            report["cads_schedule"][-1], {"time": 0.8, "anneal_time": 0.0, "gamma": 1.0}
        )
        self.assertFalse(model.pipe.transformer._forward_pre_hooks)

    def test_legacy_pool_option_preserves_identical_token_noise_draws(self):
        current, legacy = self.model(), self.model()
        legacy.cfg["cads"]["conditioning"] = "token-and-pooled-both-branches"
        validate(legacy.cfg)
        current.render("fixture", 1, 2)
        legacy.render("fixture", 1, 2)
        for a, b in zip(current.pipe.transformer.calls, legacy.pipe.transformer.calls):
            self.assertTrue(torch.equal(a["encoder_hidden_states"], b["encoder_hidden_states"]))
        self.assertFalse(
            torch.equal(
                legacy.pipe.transformer.calls[0]["pooled_projections"],
                legacy.pipe.pooled,
            )
        )

    def test_hook_is_removed_on_failure(self):
        model = self.model(fail=True)
        with self.assertRaisesRegex(RuntimeError, "Injected"):
            model.render("fixture", 1, 2)
        self.assertFalse(model.pipe.transformer._forward_pre_hooks)

    def test_sampling_progress_schedule_does_not_use_shifted_time(self):
        values = [gamma(cads_time(i, 40, 0.95, "sampling-progress"), 0.6, 0.9) for i in range(40)]
        self.assertEqual(sum(v == 0 for v in values), 4)
        self.assertEqual(sum(v == 1 for v in values), 24)
        self.assertEqual(cads_time(39, 40, 0.95, "scheduler-timestep/1000"), 0.95)
        self.assertEqual(cads_time(0, 1, 1.0, "sampling-progress"), 1.0)
        with self.assertRaises(ValueError):
            cads_time(40, 40, 0.5, "sampling-progress")

    def test_new_defaults_and_legacy_settings_are_explicit(self):
        cfg = defaults("cads")
        self.assertEqual(cfg["cads"]["conditioning"], "token-only-both-branches")
        self.assertEqual(cfg["cads"]["time"], "sampling-progress")
        for field in ["conditioning", "time"]:
            changed = copy.deepcopy(cfg)
            changed["cads"][field] = "guess"
            with self.assertRaisesRegex(ValueError, "Unsupported CADS variant"):
                validate(changed)


if __name__ == "__main__":
    unittest.main()
