# Loads in the PY_REWARD environment.
"""HPSv3 (Ma, Shui et al., 2025), a 7B VLM preference model: vendored loading and scoring.

The `hpsv3` package pins transformers==4.45.2 and pulls in deepspeed, trl, peft and fire. The
PY_REWARD environment runs transformers 5.3.0 (required by tpips) alongside the other tracked
scorers (`tracked.py`), so the inference path is vendored here instead: a Qwen2-VL backbone plus
a 3-layer regression head on the hidden state at one special token. Adapted from verl-omni's
port, `verl_omni/utils/reward_score/hpsv3_reward.py` (Apache-2.0), including its transformers-5.x
key remapping (`_remap_state_dict`).

HPSv3 has Qwen2-VL-7B read the image and the full prompt against a written rubric
(`INSTRUCTION`) and reads a score off the hidden state at a `<|Reward|>` token. It is trained on
HPDv3, 1.08M preference pairs from generators ranging from SD1.4 to GPT-Image. HPSv2 and
ImageReward truncate the prompt at 77 tokens (35 for ImageReward); HPSv3 does not, so it is the
only preference metric here that scores a 60-80 word policy prompt in full.

Scale: the head emits (mu, sigma), an uncertainty-aware Bradley-Terry parameterisation. `mu` is
the score: unbounded, centred near 0, higher is better. `score()` returns mu only.

Checkpoint: `MizzenAI/HPSv3` ships one 16.6 GB safetensors with the whole model (backbone,
vision tower, lm_head, rm_head). The Qwen2-VL-7B repo supplies only the config, tokenizer and
image processor. The checkpoint's own config.json names `Qwen2_5_VLForConditionalGeneration`,
which is wrong: the weights have Qwen2-VL's vision MLP (`visual.blocks.*.mlp.fc1/fc2`) and the
repo's HPSv3_7B.yaml names Qwen2-VL-7B-Instruct. Building from that config.json gives the
wrong architecture.
"""

import math
import os

BASE_MODEL = os.environ.get("AXIS_HPSV3_BASE", "Qwen/Qwen2-VL-7B-Instruct")
CKPT_REPO = "MizzenAI/HPSv3"
CKPT_FILE = "HPSv3.safetensors"

# The rubric the model was trained with, verbatim from hpsv3/dataset/data_collator_qwen.py.
# It is part of the metric: the head reads a hidden state conditioned on this exact text.
INSTRUCTION = (
    "\nYou are tasked with evaluating a generated image based on Visual Quality and Text"
    " Alignment and give a overall score to estimate the human preference. Please provide"
    " a rating from 0 to 10, with 0 being the worst and 10 being the best. \n"
    "\n"
    "**Visual Quality:**  \n"
    "Evaluate the overall visual quality of the image. The following sub-dimensions should be considered:\n"
    "- **Reasonableness:** The image should not contain any significant biological or logical errors,"
    " such as abnormal body structures or nonsensical environmental setups.\n"
    "- **Clarity:** Evaluate the sharpness and visibility of the image. The image should be clear and"
    " easy to interpret, with no blurring or indistinct areas.\n"
    "- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other"
    " visual elements (e.g., hair, clothing, shadows).\n"
    "- **Aesthetic and Creativity:** Assess the artistic aspects of the image, including the color"
    " scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should"
    " convey a sense of harmony and balance.\n"
    "- **Safety:** The image should not contain harmful or inappropriate content, such as political,"
    " violent, or adult material. If such content is present, the image quality and satisfaction score"
    " should be the lowest possible. \n"
    "\n"
    "**Text Alignment:**  \n"
    "Assess how well the image matches the textual prompt across the following sub-dimensions:\n"
    "- **Subject Relevance** Evaluate how accurately the subject(s) in the image (e.g., person, animal,"
    " object) align with the textual description. The subject should match the description in terms of"
    " number, appearance, and behavior.\n"
    "- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate"
    " how well the image adheres to this style.\n"
    "- **Contextual Consistency**: Assess whether the background, setting, and surrounding elements in"
    " the image logically fit the scenario described in the prompt. The environment should support and"
    " enhance the subject without contradictions.\n"
    "- **Attribute Fidelity**: Check if specific attributes mentioned in the prompt (e.g., colors,"
    " clothing, accessories, expressions, actions) are faithfully represented in the image. Minor"
    " deviations may be acceptable, but critical attributes should be preserved.\n"
    "- **Semantic Coherence**: Evaluate whether the overall meaning and intent of the prompt are"
    " captured in the image. The generated content should not introduce elements that conflict with or"
    " distort the original description.\n"
    "Textual prompt - {text_prompt}\n"
    "\n\n"
)

PROMPT_WITH_SPECIAL_TOKEN = """
Please provide the overall ratings of this image: <|Reward|>

END
"""

# HPSv3 was trained at a fixed token budget per image, so min_pixels == max_pixels ==
# 256 * 28 * 28 = 200704. `score()` resizes with PIL before the processor. Qwen2-VL-7B's
# preprocessor_config allows 3136-12845056 pixels, so the processor's own smart_resize leaves
# the image unchanged, and transformers 5.x's fast-vs-slow processor warning then concerns
# only rescale+normalize arithmetic, not resampling. Removing the pre-resize breaks this.
PIXELS = 256 * 28 * 28

_FACTOR = 28
_MAX_RATIO = 200


def _smart_resize(height, width, factor=_FACTOR, min_pixels=PIXELS, max_pixels=PIXELS):
    """Return (h, w) by Qwen2-VL's resize rule.

    Both sides are multiples of the 28px patch stride, total pixels lie in
    [min_pixels, max_pixels], and the aspect ratio is preserved. Raises ValueError if the
    aspect ratio exceeds 200.
    """
    if max(height, width) / min(height, width) > _MAX_RATIO:
        raise ValueError(f"aspect ratio must be < {_MAX_RATIO}, got {height}x{width}")
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h = math.floor(height / beta / factor) * factor
        w = math.floor(width / beta / factor) * factor
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return h, w


def checkpoint_path():
    """Return the local path to HPSv3.safetensors.

    AXIS_HPSV3_CKPT overrides. Otherwise the file comes from the HF cache, which the prefetch
    job fills because compute nodes have no outbound network.
    """
    p = os.environ.get("AXIS_HPSV3_CKPT")
    if p:
        return p
    from huggingface_hub import hf_hub_download

    return hf_hub_download(CKPT_REPO, CKPT_FILE, repo_type="model")


def _reward_model_cls():
    """Return the HPSv3 model class, defined lazily so importing this module stays cheap."""
    import torch
    import torch.nn as nn
    from transformers import Qwen2VLForConditionalGeneration

    class HPSv3RewardModel(Qwen2VLForConditionalGeneration):
        """Qwen2VLForConditionalGeneration plus `rm_head`, read at the `<|Reward|>` token.

        lm_head is kept because the checkpoint has it and the load is strict, but it is
        never called.
        """

        def __init__(self, config, output_dim=2, special_token_ids=None):
            super().__init__(config)
            self.output_dim = output_dim
            self.special_token_ids = special_token_ids
            # "ranknet" head matching the checkpoint (rm_head.0/.3/.5: 3584->1024->16->2).
            self.rm_head = nn.Sequential(
                nn.Linear(config.text_config.hidden_size, 1024),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.Linear(1024, 16),
                nn.ReLU(),
                nn.Linear(16, output_dim),
            )
            self.rm_head.to(torch.float32)

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            pixel_values=None,
            image_grid_thw=None,
            **kwargs,
        ):
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                **kwargs,
            )
            hidden = out.last_hidden_state
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                logits = self.rm_head(hidden)  # [B, L, output_dim]
            mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for tid in self.special_token_ids:
                mask |= input_ids == tid
            # One <|Reward|> token per row. A row that lost it to truncation makes view()
            # raise instead of returning a mis-pooled reward.
            return {"logits": logits[mask, ...].view(input_ids.shape[0], -1)}

    return HPSv3RewardModel


def _remap_state_dict(state_dict, model_keys):
    """Rename 4.x checkpoint keys to the transformers 5.x Qwen2-VL layout.

    transformers 5.x places the vision tower at `model.visual` and the decoder at
    `model.language_model`; the 4.x checkpoint has them at `visual` and `model`. Returns
    `state_dict` unchanged if the model does not use the 5.x layout or the keys already match.
    """
    if any("model.language_model" in k for k in model_keys) and not any(
        "language_model" in k for k in state_dict
    ):
        out = {}
        for k, v in state_dict.items():
            if "visual" in k:
                out[k.replace("visual", "model.visual")] = v
            elif "model" in k:
                out[k.replace("model", "model.language_model")] = v
            else:
                out[k] = v
        return out
    return state_dict


def load(device, ckpt=None, base_model=None):
    """Load HPSv3 and return (model, processor): ~16 GB of bf16 weights, head in fp32."""
    import safetensors.torch
    import torch
    from transformers import AutoConfig, AutoProcessor

    base_model = base_model or BASE_MODEL
    config = AutoConfig.from_pretrained(base_model)
    processor = AutoProcessor.from_pretrained(base_model, padding_side="right")
    processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<|Reward|>"]})
    special_ids = processor.tokenizer.convert_tokens_to_ids(["<|Reward|>"])

    model = _reward_model_cls()(config, output_dim=2, special_token_ids=special_ids)
    # The checkpoint's embed_tokens and lm_head are sized 151658 (151657 + the reward token),
    # not the 152064 in Qwen2-VL's config; without this the strict load fails.
    model.resize_token_embeddings(len(processor.tokenizer))
    model.to(torch.bfloat16)
    model.rm_head.to(torch.float32)
    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    sd = safetensors.torch.load_file(ckpt or checkpoint_path(), device="cpu")
    sd = sd.get("model", sd)
    model.load_state_dict(_remap_state_dict(sd, model.state_dict().keys()), strict=True)
    return model.to(device).eval(), processor


def score(model, processor, images, prompts, device):
    """Return HPSv3 mu for each (image, prompt) pair, as a list of n floats.

    Images and prompts are paired by position. `prompts` is one string applied to every image
    or a list as long as `images`.
    """
    import torch
    from PIL import Image

    if isinstance(prompts, str):
        prompts = [prompts] * len(images)
    assert len(prompts) == len(images), f"{len(prompts)} prompts for {len(images)} images"

    pil, msgs = [], []
    for im, p in zip(images, prompts):
        im = im.convert("RGB")
        h, w = _smart_resize(im.height, im.width)
        pil.append(im.resize((w, h), Image.BICUBIC))
        msgs.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": INSTRUCTION.format(text_prompt=p) + PROMPT_WITH_SPECIAL_TOKEN,
                        },
                    ],
                }
            ]
        )

    batch = processor(
        text=processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
        images=pil,
        padding=True,
        return_tensors="pt",
        videos_kwargs={"do_rescale": True},
    )
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    with torch.inference_mode():
        logits = model(return_dict=True, **batch)["logits"]
    # Column 0 is mu, column 1 is sigma.
    return [float(r[0]) for r in logits]
