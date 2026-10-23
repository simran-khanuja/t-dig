"""Explicit CLIP cosine and DINO spatial-patch scores, matching GI definitions."""


class Features:
    def __init__(self, clip_path, dino_path, device):
        from transformers import CLIPModel, AutoTokenizer, AutoModel

        self.device = device
        self.clip = CLIPModel.from_pretrained(clip_path, local_files_only=True).eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(clip_path, local_files_only=True)
        self.dino = AutoModel.from_pretrained(dino_path, local_files_only=True).eval().to(device)

    def extract(self, images, caption):
        import torch
        import torch.nn.functional as F

        image_features, patches, cls = [], [], []
        with torch.inference_mode():
            original_length = len(self.tokenizer(caption, truncation=False)["input_ids"])
            token = self.tokenizer(
                caption,
                return_tensors="pt",
                truncation=True,
                max_length=self.clip.config.text_config.max_position_embeddings,
            ).to(self.device)
            text = self.clip.text_model(**token).pooler_output
            text = F.normalize(self.clip.text_projection(text).float(), dim=-1)
            for image in images:
                # Images are B=1,C,H,W in [-1,1]; same resize/normalization as upstream.
                image = image.to(self.device).float()
                clip_pixels = (
                    F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False) + 1
                ) / 2
                mean = image.new_tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
                std = image.new_tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]
                vision = self.clip.vision_model(
                    pixel_values=(clip_pixels - mean) / std
                ).pooler_output
                image_features.append(
                    F.normalize(self.clip.visual_projection(vision).float(), dim=-1)
                )
                pixels = (
                    F.interpolate(image, size=(256, 256), mode="bilinear", align_corners=False) + 1
                ) / 2
                mean = image.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
                std = image.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
                feat = self.dino(pixel_values=(pixels - mean) / std).last_hidden_state.float()
                patches.append(F.normalize(feat[:, 1:], dim=-1))
                cls.append(F.normalize(feat[:, 0], dim=-1))
            features = torch.cat(image_features)
            patch = torch.cat(patches)
            # Dot products align the same spatial patch in both images, avoiding N*N*P*D storage.
            flattened = patch.flatten(1) / (patch.shape[1] ** 0.5)
            pair = (1 - flattened @ flattened.T).clamp(0, 2)
            pair.fill_diagonal_(0)
            return {
                "clip": (features @ text.T).flatten().cpu().numpy(),
                "clip_text_tokens_before_truncation": original_length,
                "clip_text_truncated": original_length
                > self.clip.config.text_config.max_position_embeddings,
                "pairwise": pair.cpu().numpy(),
                "dino_cls": torch.cat(cls).cpu().numpy(),
            }

    def score(self, images, caption):
        result = self.extract(images, caption)
        self.last_text_info = {
            k: result[k] for k in ["clip_text_tokens_before_truncation", "clip_text_truncated"]
        }
        return result["clip"], result["pairwise"]
