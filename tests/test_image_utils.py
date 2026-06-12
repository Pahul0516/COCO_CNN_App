import pytest
from io import BytesIO
from PIL import Image

from app.utils.image_utils import load_image_from_bytes


def _make_png(width=10, height=10, mode="RGB") -> bytes:
    img = Image.new(mode, (width, height), color=(128, 64, 32) if mode == "RGB" else (128, 64, 32, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_returns_pil_image():
    result = load_image_from_bytes(_make_png())
    assert isinstance(result, Image.Image)


def test_mode_is_rgb():
    result = load_image_from_bytes(_make_png())
    assert result.mode == "RGB"


def test_converts_rgba_to_rgb():
    result = load_image_from_bytes(_make_png(mode="RGBA"))
    assert result.mode == "RGB"


def test_preserves_dimensions():
    result = load_image_from_bytes(_make_png(width=20, height=30))
    assert result.size == (20, 30)


def test_invalid_bytes_raises():
    with pytest.raises(Exception):
        load_image_from_bytes(b"not an image")
