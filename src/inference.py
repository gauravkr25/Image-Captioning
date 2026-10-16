
from __future__ import annotations

from pathlib import Path
import time

import matplotlib
import numpy as np
import tensorflow as tf
import yaml
from PIL import Image

from src.decode import beam_search
from src.models.gru_attention import GRUDecoder
from src.models.transformer import CaptionTransformer
from src.text import Tokenizer


def load_bundle(run_name: str):
    model_dir = Path("artifacts/models") / run_name

    if not model_dir.exists():
        raise FileNotFoundError(f"Model bundle not found: {model_dir}")

    config_path = model_dir / "config.yaml"
    config = (
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config_path.exists()
        else {}
    )

    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    tokenizer = Tokenizer.load(tokenizer_path)

    encoder = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
    )
    encoder.trainable = False

    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "gru_attention")
    vocab_size = len(tokenizer.word_to_id)

    if model_type == "gru_attention":
        decoder = GRUDecoder(
            vocab_size=vocab_size,
            embedding_dim=model_cfg.get("embed_dim", 128),
            gru_units=model_cfg.get("gru_units", 256),
            feature_dim=1280,
            dropout=model_cfg.get("dropout", 0.1),
        )
    else:
        decoder = CaptionTransformer(
            vocab_size=vocab_size,
            d_model=model_cfg.get("d_model", 128),
            num_layers=model_cfg.get("num_layers", 2),
            num_heads=model_cfg.get("num_heads", 4),
            dff=model_cfg.get("dff", 512),
            dropout=model_cfg.get("dropout", 0.1),
        )

    dummy_features = tf.zeros((1, 49, 1280), dtype=tf.float32)
    dummy_ids = tf.zeros((1, 3), dtype=tf.int32)
    decoder((dummy_features, dummy_ids), training=False)

    weight_candidates = [
        model_dir / "best.weights.h5",
        model_dir / "weights.weights.h5",
        model_dir / "last.weights.h5",
    ]
    weights_path = next(
        (path for path in weight_candidates if path.exists()),
        None,
    )

    if weights_path is None:
        raise FileNotFoundError(f"No decoder weights found in {model_dir}")

    decoder.load_weights(weights_path)
    print(f"Loaded decoder weights: {weights_path}")

    return {
        "encoder": encoder,
        "decoder": decoder,
        "tokenizer": tokenizer,
        "config": config,
    }


def _encode_features(encoder, pil_image):
    image = pil_image.convert("RGB").resize(
        (224, 224),
        Image.Resampling.BILINEAR,
    )

    image_array = np.asarray(image, dtype=np.float32)
    tensor = tf.convert_to_tensor(image_array[np.newaxis, ...])

    tensor = tf.keras.applications.efficientnet_v2.preprocess_input(
        tensor
    )
    features = encoder(tensor, training=False)
    features = tf.reshape(features, (-1, features.shape[-1]))

    if features.shape[0] != 49:
        raise ValueError(
            f"Expected 49 spatial features, got {features.shape[0]}"
        )

    return features


def caption_image(
    bundle,
    pil_image,
    beam_size=3,
    return_attention=False,
):
    start_time = time.perf_counter()

    features = _encode_features(bundle["encoder"], pil_image)
    decoder = bundle["decoder"]
    tokenizer = bundle["tokenizer"]

    ranked = beam_search(
        decoder,
        features.numpy(),
        tokenizer,
        beam_size=beam_size,
        max_len=12,
        return_attention=True,
    )

    if not ranked:
        raise RuntimeError("Beam search returned no candidates.")

    best = ranked[0]
    caption, score, token_ids, attention_history = best

    # New decode.py format: each history item is (token_id, 7x7_map).
    token_attention = []
    attention_token_ids = []

    for item in attention_history:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            continue

        token_id, attention = item
        attention = np.asarray(attention, dtype=np.float32)

        if attention.shape == (7, 7):
            token_attention.append(attention)
            attention_token_ids.append(int(token_id))

    candidates = [
        (item[0], item[1], item[2])
        for item in ranked
    ]

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    result = {
        "caption": caption,
        "candidates": candidates,
        "token_attention": token_attention,
        "attention_token_ids": attention_token_ids,
        "latency_ms": latency_ms,
    }

    if return_attention:
        result["attention_history"] = attention_history

    return result


def attention_overlay(
    image,
    attn_7x7,
    alpha=0.5,
    colormap="viridis",
):
    image_arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    heat = np.asarray(attn_7x7, dtype=np.float32)

    if heat.shape != (7, 7):
        raise ValueError(f"Expected attention shape (7, 7), got {heat.shape}")

    heat_min = float(heat.min())
    heat_max = float(heat.max())

    if heat_max - heat_min < 1e-8:
        heat = np.zeros_like(heat)
    else:
        heat = (heat - heat_min) / (heat_max - heat_min)

    heat_img = Image.fromarray(
        (heat * 255).astype(np.uint8)
    ).resize(
        (image_arr.shape[1], image_arr.shape[0]),
        Image.Resampling.BILINEAR,
    )

    heat_map = np.asarray(heat_img, dtype=np.float32) / 255.0
    cmap = matplotlib.colormaps.get_cmap(colormap)
    rgba = cmap(heat_map)

    overlay = (
        image_arr * (1.0 - alpha)
        + rgba[:, :, :3] * 255.0 * alpha
    )

    return np.clip(overlay, 0, 255).astype(np.uint8)
