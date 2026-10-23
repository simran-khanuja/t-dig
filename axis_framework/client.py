"""Typed clients for the three GPU services. One place that knows each wire format."""

import urllib.parse

from . import config
from .http_util import get_json, post_json


class PolicyClient:
    def __init__(self, url=None):
        self.url = (url or config.POLICY_URL).rstrip("/")

    def health(self):
        return get_json(self.url + "/health")

    def loras(self, exp=None):
        u = self.url + "/loras"
        if exp:
            u += "?exp=" + urllib.parse.quote(exp)
        return get_json(u, timeout=30)

    def expand(self, base_prompt, axis, k=None, arm="ours", off_axis=None, **kw):
        # off_axis is left out when unset; the server treats absent and null alike.
        body = {"base_prompt": base_prompt, "axis": axis, "k": k or config.K, "arm": arm, **kw}
        if off_axis is not None:
            body["off_axis"] = float(off_axis)
        return post_json(self.url + "/expand", body, timeout=600)

    def expand_many(self, items, k=None, arm="ours", off_axis=None, **kw):
        """Expand several items in one batched call (see policy_server._generate_inputs).

        A batch takes about as long as one /expand; the 1800 s timeout covers a large K on a
        busy GPU.
        """
        body = {"items": list(items), "k": k or config.K, "arm": arm, **kw}
        if off_axis is not None:
            body["off_axis"] = float(off_axis)
        return post_json(self.url + "/expand_many", body, timeout=1800)

    def edit_plan(
        self,
        base_prompt,
        axis_a,
        axis_b,
        source_prompts,
        source_images=(),
        arm="ours",
        off_axis=None,
        exp=None,
        model=None,
        **kw
    ):
        """Return K edit instructions from the round-two policy (policy_server.route_edit_plan)."""
        body = {
            "base_prompt": base_prompt,
            "axis_a": axis_a,
            "axis_b": axis_b,
            "source_prompts": list(source_prompts),
            "source_images": list(source_images),
            "arm": arm,
            "exp": exp,
            "model": model,
            **kw,
        }
        if off_axis is not None:
            body["off_axis"] = float(off_axis)
        return post_json(self.url + "/edit_plan", body, timeout=600)


class T2IClient:
    def __init__(self, url=None):
        self.url = (url or config.T2I_URL).rstrip("/")

    def health(self):
        return get_json(self.url + "/health")

    def models(self):
        return get_json(self.url + "/models", timeout=60)

    def render(
        self,
        prompts,
        base_prompt,
        out_dir,
        tag,
        guidance_scale=None,
        init_diversity=None,
        steps=None,
        resolution=None,
        seed=None,
        model=None,
        index_offset=0,
        n_total=None,
    ):
        """Render one grid, or with index_offset/n_total one contiguous shard of it.

        A shard renders `prompts` as images [index_offset, index_offset+len) of a grid of
        `n_total`. The server slices the full grid's latents, so each image gets the noise it
        would have had in a single-server render (guarantees: t2i_server._render).
        """
        return post_json(
            self.url + "/render",
            {
                "prompts": list(prompts),
                "base_prompt": base_prompt,
                "out_dir": out_dir,
                "tag": tag,
                "model": model,
                # Unset render settings are left out; the server fills them from the model's own
                # settings.
                **({} if guidance_scale is None else {"guidance_scale": guidance_scale}),
                "init_diversity": (
                    config.INIT_DIVERSITY if init_diversity is None else init_diversity
                ),
                **({} if steps is None else {"steps": steps}),
                **({} if resolution is None else {"resolution": resolution}),
                "seed": seed,
                "index_offset": int(index_offset),
                "n_total": int(n_total or len(prompts)),
            },
        )

    def edit(
        self,
        sources,
        instructions,
        out_dir,
        tag,
        model=None,
        steps=None,
        guidance_scale=None,
        resolution=None,
        index_offset=None,
    ):
        """Apply instruction i to image i (t2i_server.route_edit)."""
        body = {
            "sources": list(sources),
            "instructions": list(instructions),
            "out_dir": out_dir,
            "tag": tag,
            "model": model,
        }
        for k, v in (
            ("steps", steps),
            ("guidance_scale", guidance_scale),
            ("resolution", resolution),
            ("index_offset", index_offset),
        ):
            if v is not None:
                body[k] = v
        return post_json(self.url + "/edit", body, timeout=1800)

    def audit(self, images, prompts, base_prompt, axis):
        return post_json(
            self.url + "/audit",
            {
                "images": list(images),
                "prompts": list(prompts),
                "base_prompt": base_prompt,
                "axis": axis,
            },
        )


class SuggestClient:
    def __init__(self, url=None):
        self.url = (url or config.SUGGEST_URL).rstrip("/")

    def health(self):
        return get_json(self.url + "/health")

    def rephrase(self, base_prompt, axis):
        return post_json(
            self.url + "/rephrase", {"base_prompt": base_prompt, "axis": axis}, timeout=60
        )

    def suggest(self, base_prompt, images, k=4, exclude=(), values=True):
        return post_json(
            self.url + "/suggest",
            {
                "base_prompt": base_prompt,
                "images": list(images),
                "k": k,
                "exclude": list(exclude),
                "values": values,
            },
            timeout=900,
        )
