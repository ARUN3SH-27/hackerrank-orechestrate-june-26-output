"""Image loading and compression for inline base64 transport to the NIM API."""
import base64
import io

import pillow_avif  # noqa: F401  (registers AVIF support with Pillow; some dataset
                     # images are AVIF/WebP/PNG despite the .jpg extension)
from PIL import Image

MAX_INLINE_BYTES = 170_000  # stay under NIM's ~180KB inline image limit
MAX_DIM = 1024


def load_and_encode(path):
    """Return (data_uri, width, height) for an image, resized/compressed to fit inline limits."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((MAX_DIM, MAX_DIM))
    quality = 85
    while True:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()
        if len(data) <= MAX_INLINE_BYTES or quality <= 30:
            break
        quality -= 15
    b64 = base64.b64encode(data).decode()
    return f"data:image/jpeg;base64,{b64}", img.width, img.height
