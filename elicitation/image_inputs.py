"""Create audited API input copies without changing original rendered images."""

from io import BytesIO
import hashlib
from PIL import Image


def resize_inputs(images, max_side):
    if max_side is None:
        return images
    if type(max_side) is not int or max_side < 1:
        raise ValueError('image_max_side must be a positive integer')
    output = []
    for source in images:
        with Image.open(BytesIO(source['data'])) as image:
            original_size = list(image.size)
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            buf = BytesIO()
            image.save(buf, format='PNG')
            data = buf.getvalue()
            sent_size = list(image.size)
        output.append(
            {
                **source,
                'data': data,
                'api_input': dict(
                    transform='Pillow thumbnail LANCZOS PNG',
                    max_side=max_side,
                    original_dimensions=original_size,
                    dimensions=sent_size,
                    sha256=hashlib.sha256(data).hexdigest(),
                    bytes=len(data),
                ),
            }
        )
    return output
