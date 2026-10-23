"""SD3.5-Large and -Medium: native Diffusers rendering, CADS hook and flow-Euler GI loop."""

from .algorithms import (
    gamma,
    cads_time,
    corrupt,
    clean_prediction,
    next_size,
    solve_subset,
)
from .assets import seed_for


class SD35:
    def __init__(self, assets, cfg):
        import torch
        import diffusers
        from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler

        if diffusers.__version__ != "0.37.0":
            raise ValueError(
                "SD3 adapter audited against diffusers 0.37.0; validate before upgrading"
            )
        self.cfg = cfg
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            assets["assets"]["renderer"]["path"],
            local_files_only=True,
            torch_dtype=getattr(torch, cfg["dtype"]),
        ).to(cfg["device"])
        if type(self.pipe.scheduler) is not FlowMatchEulerDiscreteScheduler:
            raise ValueError("Only FlowMatchEulerDiscreteScheduler is supported")
        sc = self.pipe.scheduler.config
        if sc.get("stochastic_sampling", False) or sc.get("invert_sigmas", False):
            raise ValueError("Unsupported stochastic/inverted flow scheduler")
        self.pipe.set_progress_bar_config(disable=True)
        self.features = None
        self.assets = assets

    def kwargs(self, prompt, seed):
        import torch

        c = self.cfg
        return dict(
            prompt=prompt,
            negative_prompt=c["negative_prompt"],
            height=c["resolution"],
            width=c["resolution"],
            num_inference_steps=c["steps"],
            guidance_scale=c["guidance"],
            max_sequence_length=c["max_sequence_length"],
            generator=torch.Generator("cpu").manual_seed(seed),
        )

    def render(self, prompt, seed, noise_seed=None):
        import torch

        c, trace = self.cfg, []
        handle = None
        if c["method"] == "cads":
            rng = torch.Generator("cpu").manual_seed(noise_seed)

            def hook(module, args, kwargs):
                time = float(kwargs["timestep"][0]) / self.pipe.scheduler.config.num_train_timesteps
                anneal_time = cads_time(len(trace), c["steps"], time, c["cads"]["time"])
                amount = gamma(anneal_time, c["cads"]["tau1"], c["cads"]["tau2"])
                updated = dict(kwargs)
                for field in ["encoder_hidden_states", "pooled_projections"]:
                    original = kwargs[field]
                    if original.shape[0] != 2:
                        raise ValueError(
                            "CADS requires exactly one unconditional and one conditional row"
                        )
                    changed = torch.cat(
                        [
                            corrupt(
                                branch,
                                amount,
                                c["cads"]["noise_scale"],
                                c["cads"]["psi"],
                                rng,
                            )
                            for branch in original.split(1)
                        ]
                    )
                    # Pooled projections stay clean unless conditioning is
                    # token-and-pooled-both-branches. Their noise is drawn either
                    # way, so the token-noise stream is the same in both variants.
                    if (
                        field == "encoder_hidden_states"
                        or c["cads"]["conditioning"] == "token-and-pooled-both-branches"
                    ):
                        updated[field] = changed
                trace.append({"time": time, "anneal_time": anneal_time, "gamma": amount})
                return args, updated

            handle = self.pipe.transformer.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            with torch.inference_mode():
                image = self.pipe(**self.kwargs(prompt, seed)).images[0]
            if c["method"] == "cads" and len(trace) != c["steps"]:
                raise ValueError("CADS hook did not execute exactly once per denoising step")
            return image, {
                "denoiser_candidate_steps": c["steps"],
                "cfg_branches": 2,
                "cads_schedule": trace,
            }
        finally:
            if handle is not None:
                handle.remove()

    def decode(self, latent):
        latent = latent.to(self.pipe.vae.dtype)
        latent = latent / self.pipe.vae.config.scaling_factor + self.pipe.vae.config.shift_factor
        return self.pipe.vae.decode(latent, return_dict=False)[0].clamp(-1, 1)

    def group(self, row):
        import torch
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
            retrieve_timesteps,
            calculate_shift,
        )
        from .features import Features

        if self.features is None:
            a = self.assets["assets"]
            self.features = Features(a["clip"]["path"], a["dino"]["path"], self.cfg["device"])
        p, c = self.pipe, self.cfg
        g = c["group"]
        candidate_ids = list(range(g["candidates"]))
        seeds = [seed_for(row["id"], c["seed"], i) for i in candidate_ids]
        trace, evaluations = [], 0
        with torch.inference_mode():
            pe, ne, pool, npool = p.encode_prompt(
                prompt=row["prompt"],
                prompt_2=None,
                prompt_3=None,
                negative_prompt=c["negative_prompt"],
                negative_prompt_2=None,
                negative_prompt_3=None,
                do_classifier_free_guidance=True,
                device=p._execution_device,
                num_images_per_prompt=1,
                max_sequence_length=c["max_sequence_length"],
            )
            latent = torch.cat(
                [
                    p.prepare_latents(
                        1,
                        p.transformer.config.in_channels,
                        c["resolution"],
                        c["resolution"],
                        pe.dtype,
                        p._execution_device,
                        torch.Generator("cpu").manual_seed(s),
                        None,
                    )
                    for s in seeds
                ]
            )
            scheduler_kwargs = {}
            if p.scheduler.config.get("use_dynamic_shifting", False):
                seq = (latent.shape[2] // p.transformer.config.patch_size) * (
                    latent.shape[3] // p.transformer.config.patch_size
                )
                sc = p.scheduler.config
                scheduler_kwargs["mu"] = calculate_shift(
                    seq,
                    sc.get("base_image_seq_len", 256),
                    sc.get("max_image_seq_len", 4096),
                    sc.get("base_shift", 0.5),
                    sc.get("max_shift", 1.16),
                )
            timesteps, _ = retrieve_timesteps(
                p.scheduler, c["steps"], p._execution_device, **scheduler_kwargs
            )
            for step, t in enumerate(timesteps):
                predictions = []
                cfg_active = step != 0 or g["first_step_cfg"]
                for x in latent.split(1):
                    if cfg_active:
                        v = p.transformer(
                            hidden_states=torch.cat([x, x]),
                            timestep=t.expand(2),
                            encoder_hidden_states=torch.cat([ne, pe]),
                            pooled_projections=torch.cat([npool, pool]),
                            joint_attention_kwargs=None,
                            return_dict=False,
                        )[0]
                        vu, vc = v.chunk(2)
                        v = vu + c["guidance"] * (vc - vu)
                    else:
                        v = p.transformer(
                            hidden_states=x,
                            timestep=t.expand(1),
                            encoder_hidden_states=pe,
                            pooled_projections=pool,
                            joint_attention_kwargs=None,
                            return_dict=False,
                        )[0]
                    predictions.append(v)
                velocity = torch.cat(predictions)
                evaluations += len(candidate_ids)
                x0 = clean_prediction(latent, velocity, p.scheduler.sigmas[step])
                # One stateful scheduler call per step for the whole pool.
                latent = p.scheduler.step(velocity, t, latent, return_dict=False)[0]
                if len(candidate_ids) > c["k"]:
                    count = next_size(
                        len(candidate_ids),
                        c["k"],
                        g["retention"],
                        step == len(timesteps) - 1,
                    )
                    previews = [self.decode(x) for x in x0.split(1)]
                    unary, pair = self.features.score(previews, row["prompt"])
                    keep, solver = solve_subset(
                        unary,
                        pair,
                        count,
                        g["lambda"],
                        g["solver_seconds"],
                        g["accept_incumbent"],
                    )
                    trace.append(
                        {
                            "step": step,
                            "sigma": float(p.scheduler.sigmas[step]),
                            "candidate_ids": candidate_ids,
                            "selected_ids": [candidate_ids[i] for i in keep],
                            "unary": unary.tolist(),
                            "pairwise": pair.tolist(),
                            "solver": solver,
                        }
                    )
                    trace[-1].update(self.features.last_text_info)
                    candidate_ids = [candidate_ids[i] for i in keep]
                    latent = latent[keep]
            images = [
                p.image_processor.postprocess(self.decode(x), output_type="pil")[0]
                for x in latent.split(1)
            ]
        if len(images) != c["k"]:
            raise ValueError("Group did not end at K outputs")
        return (
            images,
            candidate_ids,
            {
                "pruning": trace,
                "denoiser_candidate_steps": evaluations,
                "candidate_seeds": seeds,
                "selected_ids": candidate_ids,
                "method_variant": "sd35-corrected-flow-preview-full-vae-v1",
            },
        )
