"""PickScore v1 (Kirstain et al., NeurIPS 2023): model loading and scoring.

`tdig/paper_metrics.py` scores saved images with it: model id, normalisation and 77-token
truncation are fixed here.

The model is a CLIP ViT-H/14 finetuned on Pick-a-Pic, a set of 500k+ user preferences between
two images generated from the same prompt. The score has CLIPScore's form:

    pickscore(t, i) = logit_scale * <e_text(t), e_img(i)>      (both L2-normalised)

The score is returned raw, roughly 15-25 in practice, unbounded and not a probability. Compare
it only between images of the same prompt; its level shifts with prompt length and wording.
verl-omni's `verl_omni/utils/reward_score/pickscore_reward.py` divides the same score by 26 for
use as an RL reward, so its numbers are not comparable to these.

The CLIP text encoder truncates prompts at 77 tokens. Base prompts fit. Policy prompts are
60-80 words and do not, so `pickscore_own` scores against roughly the first 50 words. HPSv2 has
the same limit, ImageReward (BLIP) truncates at 35 tokens, and HPSv3 reads the whole prompt.
"""

import os

# PickScore ships model weights only. The processor (tokenizer and image transform) comes from
# the laion CLIP repo the weights were trained with, which HPSv2 also uses.
MODEL_ID = os.environ.get("AXIS_PICKSCORE_MODEL", "yuvalkirstain/PickScore_v1")
PROCESSOR_ID = os.environ.get("AXIS_PICKSCORE_PROCESSOR", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

# CLIP's text context length; longer prompts are truncated.
MAX_TEXT_TOKENS = 77


def load(device):
    """Load PickScore and return (model, processor), on `device` and in eval mode."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(PROCESSOR_ID)
    model = CLIPModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    return model.to(device).eval(), processor


def _features(out):
    """Return the projected embedding from a `get_*_features` output.

    transformers 4.x returns a tensor; 5.x returns a `BaseModelOutputWithPooling` with the
    projected embedding in `image_embeds` / `text_embeds`. `pooler_output` is the last resort:
    it precedes the projection head (1280-d on ViT-H, not the 1024-d shared space), so the
    caller checks that both sides have the same width.
    """
    import torch

    if isinstance(out, torch.Tensor):
        return out
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    raise TypeError(f"no embedding on CLIP output of type {type(out)!r}")


def score(model, processor, images, prompts, device):
    """Return raw PickScore for each (image, prompt) pair, as a list of n floats.

    Images and prompts are paired by position. `prompts` is one string applied to every image
    (base prompt) or a list the same length as `images` (own prompts).
    """
    import torch

    if isinstance(prompts, str):
        prompts = [prompts] * len(images)
    assert len(prompts) == len(images), f"{len(prompts)} prompts for {len(images)} images"

    img_in = processor(images=[im.convert("RGB") for im in images], return_tensors="pt")
    img_in = {k: v.to(device) for k, v in img_in.items()}
    txt_in = processor(
        text=list(prompts),
        padding=True,
        truncation=True,
        max_length=MAX_TEXT_TOKENS,
        return_tensors="pt",
    )
    txt_in = {k: v.to(device) for k, v in txt_in.items()}

    with torch.inference_mode():
        ie = _features(model.get_image_features(**img_in))
        te = _features(model.get_text_features(**txt_in))
        # Catches a `pooler_output` fallback, which would give a meaningless cosine.
        assert (
            ie.shape[-1] == te.shape[-1]
        ), f"image embeds are {ie.shape[-1]}-d, text {te.shape[-1]}-d"
        ie = ie / ie.norm(p=2, dim=-1, keepdim=True)
        te = te / te.norm(p=2, dim=-1, keepdim=True)
        s = model.logit_scale.exp() * (te * ie).sum(dim=-1)
    return [float(v) for v in s.cpu()]
