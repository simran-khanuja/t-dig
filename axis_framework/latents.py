"""Initial diversity: how correlated the K starting latents are.

    eps_k(alpha) = sqrt(1 - alpha) * eps_shared + sqrt(alpha) * eps_k

Both terms are unit-variance and independent, so the mix is standard normal and the grid does
not drift in brightness or contrast as alpha changes. alpha is the fraction of per-sample
independent noise, in [0, 1]:

  alpha = 0   one shared noise sample for all K (`render._shared_latent`), as in
              GRPO training. Variation in the grid comes from the wording only. Default.
  alpha = 1   K independent draws: images differ through the noise as well as the prompts.
              The i.i.d. stage renders here.

alpha = 0 calls `_shared_latent` directly, so for the same seed it draws exactly the latent a
training rollout drew. Latent shapes come from `render._one_latent`, which is family-aware (a
hand-built [1,C,h/8,w/8] misshapes FLUX and Qwen-Image noise).
"""

import math

# Offsets the independent draws from the shared seed. An eps_k equal to eps_shared would give
# that image variance 1 + 2*sqrt(a(1-a)) (up to 2x at alpha=0.5), a visible brightness shift.
# A large prime stride keeps every eps_k on its own seed for any k the UI allows.
_IND_STRIDE = 10_000_019
_MOD = 2**31 - 1


def _independent_latents(rs, seed, k, cfg):
    """Return K latents from K distinct seeds, via render._one_latent.

    These differ from `render_independent`'s seeds (seed, seed+1, ...), which start at the
    shared seed and would make image 0's independent draw equal the shared one.
    """
    return [rs._one_latent((int(seed) + _IND_STRIDE * (i + 1)) % _MOD, cfg) for i in range(k)]


def mixed_latents(rs, seed, k, alpha, cfg):
    """Return [k, ...] initial noise at initial-diversity *alpha* in [0, 1].

    `rs` is the `axis_framework.backend.render` module, passed in so this file imports without
    diffusers.

    Raises:
        ValueError: *alpha* outside [0, 1].
    """
    import torch

    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"init_diversity must be in [0, 1], got {alpha}")

    if alpha == 0.0:
        return rs._shared_latent(seed, k, cfg)  # same path as training

    inds = _independent_latents(rs, seed, k, cfg)
    if alpha == 1.0:
        return torch.cat(inds, dim=0)

    shared = rs._one_latent(seed, cfg)
    w_s, w_i = math.sqrt(1.0 - alpha), math.sqrt(alpha)
    # Mix in float32 and cast back: bfloat16's 8 mantissa bits would quantise the
    # interpolation into visible steps as alpha changes.
    mixed = [(w_s * shared.float() + w_i * z.float()).to(shared.dtype) for z in inds]
    return torch.cat(mixed, dim=0)
