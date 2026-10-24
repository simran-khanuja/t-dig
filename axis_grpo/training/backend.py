"""Frozen native T2I models, TPIPS, and a separate official VQAScore process."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import urllib.request

from baselines.config import t2i_model_defaults
from baselines.assets import local_checkpoint, verify_assets
from elicitation.tpips_compat import qwen_tokenizer_compat, verify_tokenizer
from .config import validate_service


class GPUBackend:
    def __init__(self, config):
        import torch
        import tpips
        from baselines.native_pool import NativeT2IModel

        self.config = validate_service(config)
        assets = verify_assets(Path(config['renderer_assets']), ['renderer'])
        cfg = t2i_model_defaults('iid', config['renderer'])
        cfg.update(config['render']['train'], device=config['device'])
        self.t2i_model = NativeT2IModel(assets, cfg)
        self.t2i_model.pipe.vae.enable_slicing()
        tpips_checkpoint = local_checkpoint(config['tpips_checkpoint'], config['cache'])
        with qwen_tokenizer_compat():
            self.tpips = tpips.load_model(
                'embedding', tpips_checkpoint, device=config['device'], cache_dir=config['cache']
            )
        verify_tokenizer(
            self.tpips,
            tpips_checkpoint,
            [
                self.tpips.create_prompt(x)
                for x in ('overall', 'chair material', 'Clock hands: 12:30. Café — 雨。')
            ],
        )
        # Editing task: the T2I model edits source images, and DreamSim measures how far each edit
        # moves from its source.
        self.dreamsim = None
        self.editor = None  # pipeline that edits; None means the generator's own pipeline
        self.img2img = None  # the editor is image-to-image (sdedit) and takes a strength
        if 'drift_weight' in config['reward']:
            # - flux2-klein-9b: the generator pipeline takes an input image and an instruction.
            # - sd35-*: no instruction editor. The image-to-image pipeline shares the generator's
            #   weights and regenerates a noised copy of the source from a description of the
            #   result.
            # - qwen-image-2512: Qwen-Image-Edit-2509, a separate checkpoint with its own
            #   transformer and Qwen2-VL processor; it shares the generator's text encoder,
            #   tokenizer, VAE and scheduler. Two 20B transformers, TPIPS and the tracked scorers
            #   do not fit on one GPU, so `_place` keeps only the transformer in use on the
            #   device. The start-up probe generates; an edit run then swaps once.
            if config['renderer'] == 'flux2-klein-9b':
                pass
            elif config['renderer'] in ('sd35-medium', 'sd35-large'):
                from diffusers import StableDiffusion3Img2ImgPipeline

                self.img2img = StableDiffusion3Img2ImgPipeline(**self.t2i_model.pipe.components)
                self.editor = self.img2img
            elif config['renderer'] == 'qwen-image-2512':
                from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
                from transformers import Qwen2VLProcessor

                parts = self.t2i_model.pipe.components
                transformer = QwenImageTransformer2DModel.from_pretrained(
                    config['editor_path'],
                    subfolder='transformer',
                    torch_dtype=parts['transformer'].dtype,
                )
                self.editor = QwenImageEditPlusPipeline(
                    scheduler=parts['scheduler'],
                    vae=parts['vae'],
                    text_encoder=parts['text_encoder'],
                    tokenizer=parts['tokenizer'],
                    transformer=transformer,
                    processor=Qwen2VLProcessor.from_pretrained(
                        config['editor_path'], subfolder='processor'
                    ),
                )
                self.editor.set_progress_bar_config(disable=True)
            else:
                raise ValueError(
                    'Image editing is wired for flux2-klein-9b, SD3.5 and qwen-image-2512 only'
                )
            from dreamsim import dreamsim

            self.dreamsim, self.dreamsim_preprocess = dreamsim(
                pretrained=True, device=config['device'], cache_dir=config['dreamsim_cache']
            )
        self.scorers = None
        if config.get('track_metrics'):
            from .tracked import TrackedScorers

            self.scorers = TrackedScorers(
                config['track_metrics'], config['device'], config.get('aesthetic_head')
            )
        self.tpips_batch = config['tpips_batch']
        self.render_batch = config['render_batch']
        self.staging = Path(config['staging']).resolve()
        self.staging.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(config['vqa_url'].rstrip('/') + '/health', timeout=30) as r:
            health = json.loads(r.read())
        if health.get('model') != config['vqa_model'] or not health.get('loaded'):
            raise RuntimeError('Wrong or unavailable VQAScore model')
        torch.cuda.synchronize()

    def _place(self, mode):
        """Move the transformer `mode` needs to the device and the other one to the host.

        `mode` is 'generate' or 'edit'. Only qwen-image-2512 has two transformers; for the
        other T2I models this does nothing.
        """
        import torch

        if self.editor is None or self.editor is self.img2img:
            return
        device = self.config['device']
        active, idle = (
            (self.editor, self.t2i_model.pipe)
            if mode == 'edit'
            else (self.t2i_model.pipe, self.editor)
        )
        if idle.transformer.device.type != 'cpu':
            idle.transformer.to('cpu')
            torch.cuda.empty_cache()
        if str(active.transformer.device) != str(torch.device(device)):
            active.transformer.to(device)

    def render(self, items):
        import torch

        results = [None] * len(items)
        for i, item in enumerate(items):
            if item.get('task') == 'edit':
                results[i] = self.edit(item)
        if any(item.get('task') != 'edit' for item in items):
            self._place('generate')
        # Training and validation settings never share a pipeline call.
        for split in ('train', 'val'):
            slots = [
                (i, j, p, item['seed'])
                for i, item in enumerate(items)
                if item['split'] == split and item.get('task') != 'edit'
                for j, p in enumerate(item['prompts'])
            ]
            self.t2i_model.cfg.update(self.config['render'][split])
            batch = (
                self.config.get('val_render_batch', self.render_batch)
                if split == 'val'
                else self.render_batch
            )
            for start in range(0, len(slots), batch):
                chunk = slots[start : start + batch]
                kwargs = self.t2i_model.kwargs([s[2] for s in chunk], chunk[0][3])
                kwargs['generator'] = [torch.Generator('cpu').manual_seed(s[3]) for s in chunk]
                kwargs['num_images_per_prompt'] = 1
                if isinstance(kwargs.get('negative_prompt'), str):
                    kwargs['negative_prompt'] = [kwargs['negative_prompt']] * len(chunk)
                with torch.inference_mode():
                    images = self.t2i_model.pipe(**kwargs).images
                if len(images) != len(chunk):
                    raise RuntimeError('T2I model returned the wrong image count')
                for (i, j, _, _), image in zip(chunk, images):
                    if results[i] is None:
                        results[i] = [None] * items[i]['k']
                    results[i][j] = image
        return results

    def sources(self, item):
        """Return the item's source images as RGB, resized to the render resolution of its split."""
        from PIL import Image

        side = self.config['render'][item['split']]['resolution']
        result = []
        for path in item['source_images']:
            with Image.open(path) as image:
                image = image.convert('RGB')
                result.append(
                    image
                    if image.size == (side, side)
                    else image.resize((side, side), Image.LANCZOS)
                )
        return result

    def edit(self, item):
        """Apply instruction i to source image i and return the edited images.

        One pipeline call per image keeps each instruction paired with its source. Every call
        uses the item's seed, as the images of a generated set do.
        """
        import torch

        self.t2i_model.cfg.update(self.config['render'][item['split']])
        self._place('edit')
        edited = []
        for source, instruction in zip(self.sources(item), item['prompts'], strict=True):
            kwargs = self.t2i_model.kwargs(instruction, item['seed'])
            kwargs.update(image=source, num_images_per_prompt=1)
            pipe = self.editor if self.editor is not None else self.t2i_model.pipe
            if self.img2img is not None:
                # Image-to-image runs steps * strength denoising steps from a noised copy of the
                # source; height and width keep the render resolution of the split.
                kwargs['strength'] = self.config.get('edit_strength', 0.6)
            with torch.inference_mode():
                edited.append(pipe(**kwargs).images[0])
        return edited

    def drift(self, item, edited):
        """Return the DreamSim distance between each source image and its edit; 0 is unchanged."""
        import torch

        device = self.config['device']
        with torch.inference_mode():
            a = torch.cat([self.dreamsim_preprocess(image) for image in self.sources(item)]).to(
                device
            )
            b = torch.cat([self.dreamsim_preprocess(image) for image in edited]).to(device)
            return [float(v) for v in self.dreamsim(a, b).float().cpu().reshape(-1)]

    def embed(self, images, factor):
        import torch

        with torch.inference_mode():
            vectors = [
                self.tpips.embed(images[i : i + self.tpips_batch], factor=factor, normalized=True)
                .float()
                .cpu()
                for i in range(0, len(images), self.tpips_batch)
            ]
        return torch.cat(vectors).numpy()

    def tracked(self, images, text):
        return self.scorers.score(images, text)

    def alignment(self, groups, items):
        pairs = []
        with tempfile.TemporaryDirectory(prefix='score-', dir=self.staging) as directory:
            for i, (images, item) in enumerate(zip(groups, items)):
                for j, image in enumerate(images):
                    pairs.append(
                        [str(Path(directory) / f'{i:04d}-{j:04d}.png'), item['base_prompt']]
                    )
            # PNG is lossless at every compression level. Level 1 on eight threads writes 16 images
            # at 1024 px in about 0.2 s, against 4.6 s at level 6, while the T2I model GPU waits.
            flat = [image for images in groups for image in images]
            with ThreadPoolExecutor(8) as pool:
                list(
                    pool.map(lambda job: job[0].save(job[1][0], compress_level=1), zip(flat, pairs))
                )
            req = urllib.request.Request(
                self.config['vqa_url'].rstrip('/') + '/score',
                data=json.dumps({'pairs': pairs}).encode(),
                headers={'Content-Type': 'application/json'},
            )
            with urllib.request.urlopen(req, timeout=self.config['timeout']) as r:
                response = json.loads(r.read())
            if response.get('model') != self.config['vqa_model'] or len(response['scores']) != len(
                pairs
            ):
                raise RuntimeError('VQAScore response identity/count mismatch')
            scores, cursor = [], 0
            for item in items:
                scores.append(response['scores'][cursor : cursor + item['k']])
                cursor += item['k']
            return scores
