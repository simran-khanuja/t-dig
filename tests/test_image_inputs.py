import hashlib
from io import BytesIO
import unittest
from PIL import Image
from elicitation.image_inputs import resize_inputs


class ResizeTests(unittest.TestCase):
    def test_copy_provenance_dimensions_and_no_upscale(self):
        buf = BytesIO()
        Image.new('RGB', (1024, 768), 'red').save(buf, format='PNG')
        raw = buf.getvalue()
        original = [
            dict(
                data=raw,
                path='original.png',
                sha256=hashlib.sha256(raw).hexdigest(),
                receipt={'seed': 42},
            )
        ]
        self.assertIs(resize_inputs(original, None), original)
        small = resize_inputs(original, 512)[0]
        self.assertEqual(small['api_input']['dimensions'], [512, 384])
        self.assertEqual(small['sha256'], original[0]['sha256'])
        self.assertEqual(small['api_input']['sha256'], hashlib.sha256(small['data']).hexdigest())
        self.assertEqual(small['receipt'], {'seed': 42})
        self.assertEqual(original[0]['data'], raw)
        self.assertEqual(resize_inputs([small], 1024)[0]['api_input']['dimensions'], [512, 384])
        self.assertEqual(resize_inputs(original, 512)[0]['data'], small['data'])
        for invalid in (True, 0, -1, '512'):
            with self.assertRaises(ValueError):
                resize_inputs(original, invalid)
