import copy
import unittest
from baselines.config import T2I_MODELS, t2i_model_defaults, validate
from baselines.native_pool import replace_methods


class T2IModelMatrixTests(unittest.TestCase):
    def test_every_default_configuration_valid(self):
        for t2i_model in T2I_MODELS:
            for method in ["iid", "group", "pag", "cads"]:
                validate(t2i_model_defaults(method, t2i_model))

    def test_distilled_model_cannot_silently_use_sd3_settings(self):
        cfg = t2i_model_defaults("group", "flux2-klein-9b")
        for key, value in [("guidance", 4.5), ("steps", 40), ("negative_prompt", "blur")]:
            broken = copy.deepcopy(cfg)
            broken[key] = value
            with self.assertRaises(ValueError):
                validate(broken)

    def test_incompatible_conditioning_is_rejected(self):
        for t2i_model in T2I_MODELS:
            cfg = t2i_model_defaults("cads", t2i_model)
            cfg["cads"]["conditioning"] = (
                "token-only-both-branches"
                if t2i_model == "flux2-klein-9b"
                else "token-only-conditional"
            )
            with self.assertRaises(ValueError):
                validate(cfg)

    def test_scoped_method_replacement_restores_on_failure(self):
        class A:
            def f(self):
                return 1

        a, b = A(), A()
        b.f = lambda: 2
        with self.assertRaises(RuntimeError):
            with replace_methods([(a, "f", lambda: 3), (b, "f", lambda: 4)]):
                self.assertEqual((a.f(), b.f()), (3, 4))
                raise RuntimeError("forced inference failure")
        self.assertNotIn("f", a.__dict__)
        self.assertEqual((a.f(), b.f()), (1, 2))


if __name__ == "__main__":
    unittest.main()
