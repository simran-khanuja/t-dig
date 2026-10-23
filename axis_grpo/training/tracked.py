"""Image-quality and text-alignment scores that are logged and never enter the reward.

`TrackedScorers` loads the requested scorers once on one device and scores the images of one
completion against the base prompt. The scorers:

- `imagereward`: ImageReward-v1.0. Unbounded; about -2 to 2. Higher is better.
- `hpsv3`: HPSv3, a Qwen2-VL-7B preference model that reads the image and the whole prompt
  against a written rubric. Unbounded, centred near 0; higher is better. The preference score
  the paper reports; the prompt is not truncated at 77 tokens. Holds about 17 GB of GPU memory.
- `hpsv2`: HPS v2.1, the scaled image-text logit of a fine-tuned CLIP ViT-H/14, for comparison
  with runs that logged it. About 0.2 to 0.35.
- `clip_score`: cosine similarity of OpenAI CLIP ViT-L/14 image and text embeddings, in [-1, 1].
  The text is truncated to CLIP's 77 tokens.
- `aesthetic`: LAION-Aesthetics V2 linear head on the same ViT-L/14 image embedding. About 1 to
  10. It ignores the prompt.

The scores stay out of the objective, so they measure image degradation independently of the
reward. A scorer failure raises rather than logging a zero.

Runs in the PY_REWARD environment, which pins `image-reward==1.5`, `hpsv2==1.2.0` and
`open_clip_torch==3.3.0` against transformers 5.x. `hpsv2.score` reloads its 4 GB checkpoint on
every call, so this module builds the same model once.
"""

from pathlib import Path

from .protocol import validate_tracked


def _patch_imagereward_blip():
    """Let ImageReward 1.5's vendored BLIP import and run under transformers 5.x.

    transformers 5.x moved three helpers out of `modeling_utils`, removed the tokenizer attributes
    `additional_special_tokens[_ids]` and the model method `get_head_mask`, and reads
    `all_tied_weights_keys` before BLIP's `init_weights()` sets it. Each is restored only when
    absent, so the patch is a no-op on a transformers release that has them.
    """
    import torch
    import transformers.modeling_utils as mu
    from transformers.pytorch_utils import apply_chunking_to_forward, prune_linear_layer
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase as tokenizer_base

    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            mask[head - sum(1 if h < head else 0 for h in already_pruned_heads)] = 0
        mask = mask.view(-1).contiguous().eq(1)
        return heads, torch.arange(len(mask))[mask].long()

    for name, value in (
        ("apply_chunking_to_forward", apply_chunking_to_forward),
        ("prune_linear_layer", prune_linear_layer),
        ("find_pruneable_heads_and_indices", find_pruneable_heads_and_indices),
    ):
        if not hasattr(mu, name):
            setattr(mu, name, value)
    if not hasattr(tokenizer_base, "additional_special_tokens"):

        def additional(self):
            named = {str(v) for v in self.special_tokens_map.values()}
            return [t for t in self.all_special_tokens if t not in named]

        tokenizer_base.additional_special_tokens = property(additional)
    if not hasattr(tokenizer_base, "additional_special_tokens_ids"):
        tokenizer_base.additional_special_tokens_ids = property(
            lambda self: self.convert_tokens_to_ids(self.additional_special_tokens)
        )
    if not hasattr(mu.PreTrainedModel, "all_tied_weights_keys"):
        key = "_tdig_all_tied_weights_keys"

        def get_tied(self):
            value = self.__dict__.get(key)
            if value is None:
                try:
                    value = self.get_expanded_tied_weights_keys(all_submodels=False)
                except Exception:
                    value = {}
                self.__dict__[key] = value
            return value

        mu.PreTrainedModel.all_tied_weights_keys = property(
            get_tied, lambda self, value: self.__dict__.__setitem__(key, value)
        )
    if not hasattr(mu.PreTrainedModel, "get_head_mask"):

        def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is None:
                return [None] * num_hidden_layers
            if head_mask.dim() == 1:
                head_mask = head_mask[None, None, :, None, None].expand(
                    num_hidden_layers, -1, -1, -1, -1
                )
            elif head_mask.dim() == 2:
                head_mask = head_mask[:, None, :, None, None]
            head_mask = head_mask.to(dtype=self.dtype)
            return head_mask.unsqueeze(-1) if is_attention_chunked else head_mask

        mu.PreTrainedModel.get_head_mask = get_head_mask


class TrackedScorers:
    def __init__(self, names, device, aesthetic_head=None):
        """Load the requested scorers.

        Args:
            names: subset of `TRACKED`.
            device: torch device string for every scorer.
            aesthetic_head: path to `sac_logos_ava1_l14_linearMSE.pth`. Required when
                `aesthetic` is requested.

        Weights come from the Hugging Face hub cache named by `HF_HUB_CACHE`
        (`xswu/HPSv2`, the two CLIP backbones) and from `~/.cache/ImageReward`. Raises
        `ValueError` for an unknown name and `FileNotFoundError` for a missing aesthetic head.
        """
        import torch

        self.names = validate_tracked(names)
        self.device = device
        if "imagereward" in self.names:
            _patch_imagereward_blip()
            import ImageReward

            self.imagereward = ImageReward.load("ImageReward-v1.0", device=device)
        if "hpsv3" in self.names:
            from . import hpsv3_model

            self.hpsv3_model = hpsv3_model
            self.hps3, self.hps3_processor = hpsv3_model.load(device)
        if "hpsv2" in self.names:
            import huggingface_hub
            from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
            from hpsv2.utils import hps_version_map

            # The argument list is the one `hpsv2.img_score.initialize_model` uses.
            model, _, self.hps_preprocess = create_model_and_transforms(
                "ViT-H-14",
                "laion2B-s32B-b79K",
                precision="amp",
                device=device,
                jit=False,
                force_quick_gelu=False,
                force_custom_text=False,
                force_patch_dropout=False,
                force_image_size=None,
                pretrained_image=False,
                image_mean=None,
                image_std=None,
                light_augmentation=True,
                aug_cfg={},
                output_dict=True,
                with_score_predictor=False,
                with_region_predictor=False,
            )
            checkpoint = torch.load(
                huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map["v2.1"]),
                map_location=device,
            )
            model.load_state_dict(checkpoint["state_dict"])
            self.hps, self.hps_tokenizer = model.to(device).eval(), get_tokenizer("ViT-H-14")
        if {"clip_score", "aesthetic"} & set(self.names):
            import open_clip

            clip, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="openai"
            )
            self.clip, self.clip_tokenizer = clip.to(device).eval(), open_clip.get_tokenizer(
                "ViT-L-14"
            )
        if "aesthetic" in self.names:
            if not aesthetic_head or not Path(aesthetic_head).is_file():
                raise FileNotFoundError(f"LAION aesthetic head not found: {aesthetic_head}")
            layers = torch.nn.Sequential(
                torch.nn.Linear(768, 1024),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(1024, 128),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(128, 64),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(64, 16),
                torch.nn.Linear(16, 1),
            )
            state = torch.load(aesthetic_head, map_location="cpu")
            layers.load_state_dict(
                {k.replace("layers.", ""): v for k, v in state.items()}, strict=True
            )
            self.aesthetic = layers.to(device).eval()

    def score(self, images, text):
        """Score each image against one text.

        Args:
            images: list of PIL images.
            text: the base prompt.

        Returns `{name: [float per image]}` for every requested scorer. Raises `RuntimeError`
        when a scorer returns a non-finite value.
        """
        import math
        import torch

        result = {}
        with torch.inference_mode():
            if "imagereward" in self.names:
                result["imagereward"] = [
                    float(self.imagereward.score(text, [image])) for image in images
                ]
            if "hpsv3" in self.names:
                result["hpsv3"] = [
                    float(v)
                    for v in self.hpsv3_model.score(
                        self.hps3, self.hps3_processor, images, [text] * len(images), self.device
                    )
                ]
            if "hpsv2" in self.names:
                pixels = torch.stack([self.hps_preprocess(image) for image in images]).to(
                    self.device
                )
                tokens = self.hps_tokenizer([text]).to(self.device)
                with torch.autocast(device_type=torch.device(self.device).type):
                    out = self.hps(pixels, tokens)
                    logits = out["image_features"] @ out["text_features"].T
                result["hpsv2"] = logits[:, 0].float().cpu().tolist()
            if hasattr(self, "clip"):
                pixels = torch.stack(
                    [self.clip_preprocess(image.convert("RGB")) for image in images]
                ).to(self.device)
                embedding = self.clip.encode_image(pixels).float()
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                if "clip_score" in self.names:
                    words = self.clip.encode_text(
                        self.clip_tokenizer([text]).to(self.device)
                    ).float()
                    words = words / words.norm(dim=-1, keepdim=True)
                    result["clip_score"] = (embedding @ words.T)[:, 0].cpu().tolist()
                if "aesthetic" in self.names:
                    result["aesthetic"] = self.aesthetic(embedding).squeeze(-1).cpu().tolist()
        for name, values in result.items():
            if len(values) != len(images) or not all(math.isfinite(v) for v in values):
                raise RuntimeError(f"Tracked scorer {name} returned an invalid value")
        return result
