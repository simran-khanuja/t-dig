"""Group Inference and CADS on the pinned native Diffusers denoising loop.

The pipeline holds one placeholder latent. The transformer evaluates each
candidate separately, and the pipeline combines the results with its own CFG
(including Qwen's norm correction). The scheduler advances the whole pool once
per step, the pool is pruned, and one placeholder returns to the loop. There is
no prompt batching, scheduler reimplementation or cross-candidate attention.
"""

import inspect
from contextlib import contextmanager

from .algorithms import gamma, cads_time, corrupt, clean_prediction, next_size, solve_subset
from .assets import seed_for


@contextmanager
def replace_methods(replacements):
    """Set attributes for the duration of the block; restore instance and inherited ones after."""
    saved = []
    try:
        for obj, name, value in replacements:
            saved.append((obj, name, name in obj.__dict__, obj.__dict__.get(name)))
            setattr(obj, name, value)
        yield
    finally:
        for obj, name, existed, value in reversed(saved):
            if existed:
                setattr(obj, name, value)
            else:
                delattr(obj, name)


class NativeT2IModel:
    def __init__(self, assets, cfg):
        import torch
        import diffusers
        from diffusers import (
            Flux2KleinPipeline,
            QwenImagePipeline,
            StableDiffusion3Pipeline,
            FlowMatchEulerDiscreteScheduler,
        )

        if diffusers.__version__ != "0.37.0":
            raise ValueError("Native pool adapter requires audited diffusers 0.37.0")
        self.cfg, self.assets, self.features = cfg, assets, None
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cls = {
            "flux2-klein-9b": Flux2KleinPipeline,
            "qwen-image-2512": QwenImagePipeline,
            "sd35-medium": StableDiffusion3Pipeline,
            "sd35-large": StableDiffusion3Pipeline,
        }[cfg["renderer"]]
        self.pipe = cls.from_pretrained(
            assets["assets"]["renderer"]["path"],
            local_files_only=True,
            torch_dtype=getattr(torch, cfg["dtype"]),
        ).to(cfg["device"])
        if type(self.pipe.scheduler) is not FlowMatchEulerDiscreteScheduler:
            raise ValueError("Require native flow Euler scheduler")
        sc = self.pipe.scheduler.config
        if sc.get("stochastic_sampling", False) or sc.get("invert_sigmas", False):
            raise ValueError("Unsupported flow scheduler")
        if self.is_flux and self.pipe.config.is_distilled is not True:
            raise ValueError("Expected distilled Klein checkpoint")
        self.pipe.set_progress_bar_config(disable=True)

    @property
    def is_flux(self):
        return self.cfg["renderer"] == "flux2-klein-9b"

    @property
    def is_sd3(self):
        return self.cfg["renderer"].startswith("sd35")

    def kwargs(self, prompt, seed):
        import torch

        c = self.cfg
        result = dict(
            prompt=prompt,
            height=c["resolution"],
            width=c["resolution"],
            num_inference_steps=c["steps"],
            max_sequence_length=c["max_sequence_length"],
            generator=torch.Generator("cpu").manual_seed(seed),
        )
        if self.is_sd3:
            result.update(guidance_scale=c["guidance"], negative_prompt=c["negative_prompt"])
        elif self.is_flux:
            result["guidance_scale"] = c["guidance"]
        else:
            result.update(true_cfg_scale=c["guidance"], negative_prompt=c["negative_prompt"])
        return result

    def decode(self, latent, latent_ids=None):
        import torch

        p = self.pipe
        if self.is_sd3:
            latent = (
                latent.to(p.vae.dtype) / p.vae.config.scaling_factor + p.vae.config.shift_factor
            )
            image = p.vae.decode(latent, return_dict=False)[0]
        elif self.is_flux:
            # Clean previews are computed in float32; the native VAE is bf16.
            latent = latent.to(p.vae.dtype)
            latent = p._unpack_latents_with_ids(latent, latent_ids)
            mean = p.vae.bn.running_mean.view(1, -1, 1, 1).to(latent)
            std = (
                (p.vae.bn.running_var.view(1, -1, 1, 1) + p.vae.config.batch_norm_eps)
                .sqrt()
                .to(latent)
            )
            latent = p._unpatchify_latents(latent * std + mean)
            image = p.vae.decode(latent, return_dict=False)[0]
        else:
            r = self.cfg["resolution"]
            latent = p._unpack_latents(latent, r, r, p.vae_scale_factor).to(p.vae.dtype)
            mean = (
                torch.tensor(p.vae.config.latents_mean)
                .view(1, p.vae.config.z_dim, 1, 1, 1)
                .to(latent)
            )
            inv_std = 1.0 / torch.tensor(p.vae.config.latents_std).view(
                1, p.vae.config.z_dim, 1, 1, 1
            ).to(latent)
            image = p.vae.decode(latent / inv_std + mean, return_dict=False)[0][:, :, 0]
        return image.clamp(-1, 1)

    def render(self, prompt, seed, noise_seed=None):
        import torch

        c, p, trace = self.cfg, self.pipe, []
        handle = None
        branches = 1 if self.is_flux else 2
        if c["method"] == "cads":
            rng = torch.Generator("cpu").manual_seed(noise_seed)

            def hook(module, args, kwargs):
                # Both native pipelines call the transformer once per CFG branch.
                time = float(kwargs["timestep"][0])
                step = len(trace) // branches
                anneal = cads_time(step, c["steps"], time, c["cads"]["time"])
                amount = gamma(anneal, c["cads"]["tau1"], c["cads"]["tau2"])
                original = kwargs["encoder_hidden_states"]
                if original.shape[0] != 1:
                    raise ValueError("Native CADS expects one prompt per call")
                mask = kwargs.get("encoder_hidden_states_mask")
                # Leave padding unchanged; rescale statistics use valid tokens only.
                changed = original.clone()
                if mask is None:
                    changed = corrupt(
                        original, amount, c["cads"]["noise_scale"], c["cads"]["psi"], rng
                    )
                else:
                    valid = mask[0].bool()
                    changed[:, valid] = corrupt(
                        original[:, valid], amount, c["cads"]["noise_scale"], c["cads"]["psi"], rng
                    )
                trace.append(dict(step=step, branch=len(trace) % branches, time=time, gamma=amount))
                return args, {**kwargs, "encoder_hidden_states": changed}

            handle = p.transformer.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            with torch.no_grad():
                image = p(**self.kwargs(prompt, seed)).images[0]
            if handle is not None and len(trace) != c["steps"] * branches:
                raise ValueError("Unexpected CADS branch/step count")
            return image, dict(
                denoiser_candidate_steps=c["steps"], cfg_branches=branches, cads_schedule=trace
            )
        finally:
            if handle is not None:
                handle.remove()

    def group(self, row):
        import torch
        from .features import Features

        c, p = self.cfg, self.pipe
        g = c["group"]
        ids = list(range(g["candidates"]))
        seeds = [seed_for(row["id"], c["seed"], i) for i in ids]
        if len(ids) > c["k"] and self.features is None:
            a = self.assets["assets"]
            self.features = Features(a["clip"]["path"], a["dino"]["path"], c["device"])
        original_prepare, original_forward, original_step = (
            p.prepare_latents,
            p.transformer.forward,
            p.scheduler.step,
        )
        latent, latent_ids, trace, evaluations, step = None, None, [], 0, 0

        def prepare(*args, **kwargs):
            nonlocal latent, latent_ids
            bound = inspect.signature(original_prepare).bind(*args, **kwargs)
            if bound.arguments["batch_size"] != 1 or bound.arguments.get("latents") is not None:
                raise ValueError("Native pool requires one prompt and fresh latents")
            values = []
            first = None
            for seed in seeds:
                bound.arguments["generator"] = torch.Generator("cpu").manual_seed(seed)
                result = original_prepare(*bound.args, **bound.kwargs)
                if first is None:
                    first = result
                values.append(result[0] if self.is_flux else result)
            latent = torch.cat(values)
            if self.is_flux:
                latent_ids = first[1]
            return first

        def predict_call(pool, kwargs):
            if kwargs["hidden_states"].shape[0] != 1:
                raise ValueError("Native pool transformer contract changed")
            dtype = kwargs["hidden_states"].dtype
            return torch.cat(
                [
                    original_forward(**{**kwargs, "hidden_states": value.to(dtype)})[0]
                    for value in pool.split(1)
                ]
            )

        def forward(*args, **kwargs):
            if args:
                raise ValueError("Unexpected positional transformer arguments")
            return (predict_call(latent, kwargs),)

        def advance(model_output, timestep, sample, **kwargs):
            nonlocal latent, ids, evaluations, step
            if (
                sample.shape[0] != 1
                or model_output.shape != latent.shape
                or kwargs != {"return_dict": False}
            ):
                raise ValueError("Native pool scheduler contract changed")
            sigma = p.scheduler.sigmas[step]
            x0 = clean_prediction(latent, model_output, sigma)
            latent = original_step(model_output, timestep, latent, **kwargs)[0]
            evaluations += len(ids)
            if len(ids) > c["k"]:
                n = next_size(len(ids), c["k"], g["retention"], step == c["steps"] - 1)
                previews = [self.decode(x, latent_ids) for x in x0.split(1)]
                unary, pair = self.features.score(previews, row["prompt"])
                keep, solver = solve_subset(
                    unary, pair, n, g["lambda"], g["solver_seconds"], g["accept_incumbent"]
                )
                trace.append(
                    dict(
                        step=step,
                        sigma=float(sigma),
                        candidate_ids=ids,
                        selected_ids=[ids[i] for i in keep],
                        unary=unary.tolist(),
                        pairwise=pair.tolist(),
                        solver=solver,
                        **self.features.last_text_info
                    )
                )
                ids = [ids[i] for i in keep]
                latent = latent[keep]
            step += 1
            return (latent[:1],)

        with torch.no_grad(), replace_methods(
            [
                (p, "prepare_latents", prepare),
                (p.transformer, "forward", forward),
                (p.scheduler, "step", advance),
            ]
        ):
            p(**self.kwargs(row["prompt"], seeds[0]), output_type="latent")
            if step != c["steps"] or len(ids) != c["k"]:
                raise ValueError("Native pool did not finish at K with the expected steps")
            images = [
                p.image_processor.postprocess(self.decode(x, latent_ids), output_type="pil")[0]
                for x in latent.split(1)
            ]
        return (
            images,
            ids,
            dict(
                pruning=trace,
                candidate_seeds=seeds,
                selected_ids=ids,
                denoiser_candidate_steps=evaluations,
                method_variant="native-flow-pool-v1",
            ),
        )
