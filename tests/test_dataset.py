import json

import numpy as np
import tensorflow as tf

from src.dataset import build_caption_table, make_tf_dataset, split_image_captions
from src.text import Tokenizer
from src.train import _load_real_split


def test_make_tf_dataset_shifts_decoder_input_and_target(synthetic_features_and_captions):
    features, captions = synthetic_features_and_captions
    tokenizer = Tokenizer(min_word_freq=1)
    tokenizer.fit(captions)
    tokenizer.max_len = 10

    dataset = make_tf_dataset("train", tokenizer, (features, captions), batch_size=2, training=True, max_len=10)
    batch = next(iter(dataset))
    (feature_batch, decoder_input), target = batch

    assert feature_batch.shape[0] == 2
    assert decoder_input.shape[1] == 9
    assert target.shape == decoder_input.shape
    assert decoder_input.dtype == tf.int32
    assert target.dtype == tf.int32


def test_build_caption_table_and_split_by_image(tmp_path):
    image_dir = tmp_path / "Images"
    image_dir.mkdir()
    (image_dir / "img1.jpg").touch()
    (image_dir / "img2.jpg").touch()
    (image_dir / "img3.jpg").touch()
    captions_file = tmp_path / "Flickr8k.token.txt"
    captions_file.write_text(
        "img1.jpg#0\ta dog runs in grass\n"
        "img1.jpg#1\ta dog sits by a tree\n"
        "img2.jpg#0\ta person holds a ball\n"
        "img2.jpg#1\ta person walks outside\n"
        "img3.jpg#0\ttwo cats stare at the sky\n",
        encoding="utf-8",
    )

    caption_table = build_caption_table(tmp_path)
    assert set(caption_table["image"]) == {"img1.jpg", "img2.jpg", "img3.jpg"}

    splits = split_image_captions(caption_table, val_fraction=0.2, test_fraction=0.2, seed=7)
    assert set(splits).issuperset({"train", "val", "test"})
    assert set(splits["train"]["image"]).union(splits["val"]["image"]).union(splits["test"]["image"]) == {"img1.jpg", "img2.jpg", "img3.jpg"}
    assert set(splits["train"]["image"]).isdisjoint(splits["val"]["image"]) is True
    assert set(splits["train"]["image"]).isdisjoint(splits["test"]["image"]) is True
    assert set(splits["val"]["image"]).isdisjoint(splits["test"]["image"]) is True


def test_load_real_split_reads_cached_feature_bundle(monkeypatch, tmp_path):
    root = tmp_path / "dataset"
    images_dir = root / "Images"
    images_dir.mkdir(parents=True)
    (images_dir / "img1.jpg").write_bytes(b"fake")
    (images_dir / "img2.jpg").write_bytes(b"fake")
    (root / "Flickr8k.token.txt").write_text(
        "img1.jpg#0\ta dog runs in grass\n"
        "img1.jpg#1\ta dog sits near a tree\n"
        "img2.jpg#0\ta cat watches the sky\n",
        encoding="utf-8",
    )

    feature_dir = tmp_path / "artifacts" / "features" / "EfficientNetV2B0"
    feature_dir.mkdir(parents=True)
    feature_array = np.zeros((3, 49, 1280), dtype=np.float32)
    np.save(feature_dir / "train.npy", feature_array)
    (feature_dir / "train_index.json").write_text(json.dumps({"img1.jpg": 0, "img2.jpg": 2}), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    cfg = {"data": {"root": str(root), "val_fraction": 0.1, "test_fraction": 0.1}, "seed": 42, "encoder": {"backbone": "EfficientNetV2B0"}}
    features, captions = _load_real_split(cfg, "train")

    assert features.shape[0] == len(captions)
    assert captions == [
        "a dog runs in grass",
        "a dog sits near a tree",
        "a cat watches the sky",
    ]
