import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from axis_grpo.training.image_data import image_record, image_row


def record(root):
    images = []
    for i in range(4):
        p = root / f'source{i}.png'
        Image.new('RGB', (1024, 1024), (i * 50, 30, 70)).save(p)
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        receipt = dict(prompt_id='id', index=i, render_prompt='A tree', sha256=h)
        images.append(dict(path=str(p), sha256=h, receipt=receipt))
    return dict(row=dict(id='id', prompt='A tree', renderer='qwen-image-2512'), images=images)


class ImageDataTests(unittest.TestCase):
    def test_resize_resume_and_source_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            r = record(root)
            original = [Path(im['path']).read_bytes() for im in r['images']]
            saved = image_record(r, root / 'cache')
            self.assertEqual(saved, image_record(r, root / 'cache'))
            self.assertEqual(original, [Path(im['path']).read_bytes() for im in r['images']])
            for im in saved['images']:
                with Image.open(im['path']) as image:
                    self.assertEqual(image.size, (512, 512))
            bad = copy.deepcopy(r)
            bad['images'][1]['receipt']['index'] = 2
            with self.assertRaises(ValueError):
                image_record(bad, root / 'other-cache')
            Path(saved['images'][0]['path']).write_bytes(b'bad')
            with self.assertRaises(ValueError):
                image_record(r, root / 'cache')

    def test_row_keeps_prompt_reward_and_extra_info(self):
        images = dict(
            identity=dict(prompt_id='id', renderer='qwen-image-2512', prompt='A tree'),
            images=[dict(path=str(i), sha256=str(i)) for i in range(4)],
        )
        row = dict(
            prompt=[dict(role='user', content='text instruction')],
            reward_model=dict(ground_truth='unchanged'),
            extra_info=dict(
                prompt_id='id',
                source_renderer='qwen-image-2512',
                base_prompt='A tree',
                axis='branch shape',
                k=4,
                row_id='r',
            ),
        )
        original = copy.deepcopy(row)
        output = image_row(row, images)
        self.assertEqual(row, original)
        self.assertEqual(output['prompt'], row['prompt'])
        self.assertEqual(output['reward_model'], row['reward_model'])
        self.assertEqual(output['images'], ['0', '1', '2', '3'])
        self.assertEqual(output['extra_info']['input_image_sha256'], ['0', '1', '2', '3'])
        for key, value in row['extra_info'].items():
            self.assertEqual(output['extra_info'][key], value)
        images['identity']['renderer'] = 'flux2-klein-9b'
        with self.assertRaises(ValueError):
            image_row(row, images)


if __name__ == '__main__':
    unittest.main()
