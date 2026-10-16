import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.features import extract_features


def test_extract_features_writes_npy_and_index(tmp_path):
    root = tmp_path / "data"
    image_dir = root / "Images"
    image_dir.mkdir(parents=True)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    Image.fromarray(image).save(image_dir / "sample.png")
    cfg = {"data": {"root": str(root)}, "encoder": {"backbone": "EfficientNetV2B0"}}
    result = extract_features(cfg, "train")
    feature_path = Path("artifacts/features/EfficientNetV2B0/train.npy")
    assert feature_path.exists()
    feature_array = np.load(feature_path, allow_pickle=True)
    assert feature_array.shape[1:] == (49, 1280)
    assert feature_array.dtype == np.float16
    index_path = Path("artifacts/features/EfficientNetV2B0/train_index.json")
    assert index_path.exists()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert "sample.png" in payload
    assert result["features"].shape[1:] == (49, 1280)


def test_extract_features_generates_synthetic_smoke_images_when_missing(tmp_path):
    root = tmp_path / "data"
    cfg = {"data": {"root": str(root), "smoke": True}, "encoder": {"backbone": "EfficientNetV2B0"}}
    result = extract_features(cfg, "train")
    image_dir = root / "Images"
    assert image_dir.exists()
    assert len(list(image_dir.glob("*.png"))) >= 1
    assert result["features"].shape[1:] == (49, 1280)
    assert result["index"]
