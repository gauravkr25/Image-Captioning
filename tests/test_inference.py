import numpy as np
from PIL import Image

from src.inference import attention_overlay, caption_image


def test_attention_overlay_returns_image_like_array():
    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    overlay = attention_overlay(image, np.ones((7, 7), dtype=np.float32), alpha=0.5)
    assert overlay.shape == (32, 32, 3)
    assert overlay.dtype == np.uint8


def test_caption_image_handles_placeholder_bundle():
    bundle = {"encoder": None, "decoder": None, "tokenizer": None, "config": {}}
    image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    try:
        caption_image(bundle, image)
    except Exception as exc:
        assert isinstance(exc, TypeError) or isinstance(exc, AttributeError)
