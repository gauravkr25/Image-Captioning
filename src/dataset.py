from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


def _parse_image_name(raw_value) -> str:
    if raw_value is None:
        return ""
    value = str(raw_value).strip()
    if not value:
        return ""
    value = value.split("#", 1)[0].strip()
    value = value.replace("\\", "/")
    value = value.split("\t", 1)[0].strip()
    value = value.split(",", 1)[0].strip()
    value = value.split(" ", 1)[0].strip() if value and " " in value and value.split()[0] not in {"", " "} else value
    if "/" in value:
        value = Path(value).name
    return value


def _read_official_split(path: str | Path) -> set[str]:
    split_path = Path(path)
    if not split_path.exists():
        return set()

    names: set[str] = set()
    with split_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            image_name = _parse_image_name(line)
            if image_name:
                names.add(image_name)
    return names


def _load_official_splits(root: str | Path | None) -> dict[str, set[str]]:
    if root is None:
        return {}

    root_dir = Path(root)
    if not root_dir.exists():
        return {}

    filenames_by_split = {
        "train": [
            "train.txt",
            "train_images.txt",
            "Flickr8k.trainImages.txt",
            "flickr8k.trainImages.txt",
            "Flickr_8k.trainImages.txt",
            "flickr_8k.trainImages.txt",
        ],
        "val": [
            "val.txt",
            "dev.txt",
            "dev_images.txt",
            "Flickr8k.devImages.txt",
            "flickr8k.devImages.txt",
            "Flickr_8k.devImages.txt",
            "flickr_8k.devImages.txt",
        ],
        "test": [
            "test.txt",
            "test_images.txt",
            "Flickr8k.testImages.txt",
            "flickr8k.testImages.txt",
            "Flickr_8k.testImages.txt",
            "flickr_8k.testImages.txt",
        ],
    }

    official = {}
    for split_name, filenames in filenames_by_split.items():
        names: set[str] = set()
        for filename in filenames:
            names |= _read_official_split(root_dir / filename)
        if names:
            official[split_name] = names
    return official


def build_caption_table(root):
    root = Path(root)
    for filename in ["captions.txt", "Flickr8k.token.txt", "flickr8k.token.txt"]:
        captions_file = root / filename
        if captions_file.exists():
            if filename == "captions.txt":
                df = pd.read_csv(captions_file)
                if {"image", "caption"}.issubset(df.columns):
                    return df[["image", "caption"]].copy()
            rows = []
            with captions_file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    if "\t" in line:
                        image_token, caption = line.split("\t", 1)
                    elif " " in line:
                        image_token, caption = line.split(" ", 1)
                    else:
                        continue
                    image_name = image_token.split("#", 1)[0]
                    rows.append({"image": image_name, "caption": caption.strip()})
            if rows:
                return pd.DataFrame(rows)
    image_dir = root / "Images"
    if image_dir.exists():
        images = sorted(image_dir.glob("*.*"))
        rows = [{"image": img.name, "caption": ""} for img in images]
        return pd.DataFrame(rows)
    raise FileNotFoundError(f"No caption file or Images directory found under {root}")


def _mask_split_images(caption_table, split_names):
    if not split_names:
        return caption_table.iloc[0:0].copy()
    normalized = {str(_parse_image_name(name)) for name in split_names if _parse_image_name(name)}
    return caption_table[caption_table["image"].map(lambda image_name: str(_parse_image_name(image_name)) in normalized)].copy()


def split_image_captions(caption_table, val_fraction=0.1, test_fraction=0.1, seed=42):
    if not isinstance(caption_table, pd.DataFrame):
        raise TypeError("caption_table must be a pandas DataFrame")
    if {"image", "caption"}.difference(caption_table.columns):
        raise ValueError("caption_table must contain 'image' and 'caption' columns")

    val_fraction = float(val_fraction)
    test_fraction = float(test_fraction)
    val_fraction = max(0.0, min(val_fraction, 0.5))
    test_fraction = max(0.0, min(test_fraction, 0.5))

    official = _load_official_splits(Path("."))
    if official:
        relevant = {split_name: _mask_split_images(caption_table, split_names) for split_name, split_names in official.items()}
        if set(relevant).issuperset({"train", "val", "test"}):
            for split_name, split_table in relevant.items():
                if split_name not in {"train", "val", "test"}:
                    continue
            return {"train": relevant.get("train", caption_table.iloc[0:0].copy()), "val": relevant.get("val", caption_table.iloc[0:0].copy()), "test": relevant.get("test", caption_table.iloc[0:0].copy())}

    images = sorted({str(_parse_image_name(image_name)) for image_name in caption_table["image"].dropna() if _parse_image_name(image_name)})
    if len(images) < 3:
        return {"train": caption_table.copy(), "val": caption_table.iloc[0:0].copy(), "test": caption_table.iloc[0:0].copy()}
    if val_fraction == 0.0 and test_fraction == 0.0:
        return {"train": caption_table.copy(), "val": caption_table.iloc[0:0].copy(), "test": caption_table.iloc[0:0].copy()}

    rng = np.random.default_rng(seed)
    shuffled = images.copy()
    rng.shuffle(shuffled)
    n_images = len(shuffled)

    if val_fraction > 0.0 and test_fraction > 0.0:
        val_count = max(1, int(round(n_images * val_fraction)))
        test_count = max(1, int(round(n_images * test_fraction)))
        if val_count + test_count >= n_images:
            val_count = max(1, n_images // 3)
            test_count = max(1, n_images - val_count - 1)
        test_images = set(shuffled[:test_count])
        remaining = shuffled[test_count:]
        val_images = set(remaining[:val_count])
        train_images = set(remaining[val_count:])
    elif val_fraction > 0.0:
        val_count = max(1, int(round(n_images * val_fraction)))
        if val_count >= n_images:
            val_count = max(1, n_images - 1)
        val_images = set(shuffled[:val_count])
        train_images = set(shuffled[val_count:])
        test_images = set()
    else:
        test_count = max(1, int(round(n_images * test_fraction)))
        if test_count >= n_images:
            test_count = max(1, n_images - 1)
        test_images = set(shuffled[:test_count])
        train_images = set(shuffled[test_count:])
        val_images = set()

    splits = {
        "train": caption_table[caption_table["image"].map(lambda image_name: str(_parse_image_name(image_name)) in train_images)].copy(),
        "val": caption_table[caption_table["image"].map(lambda image_name: str(_parse_image_name(image_name)) in val_images)].copy(),
        "test": caption_table[caption_table["image"].map(lambda image_name: str(_parse_image_name(image_name)) in test_images)].copy(),
    }
    return splits


def make_tf_dataset(split, tokenizer, features, batch_size, training=True, max_len=None):
    if isinstance(features, dict):
        feat_array = features["features"]
        captions = features["captions"]
    elif isinstance(features, tuple) and len(features) == 2:
        feat_array, captions = features
    else:
        feat_array = features
        captions = ["" for _ in range(len(feat_array))]

    max_len = max_len or getattr(tokenizer, "max_len", 16)
    sequences = [tokenizer.encode(caption, max_len=max_len) for caption in captions]
    seq_array = tf.convert_to_tensor(np.asarray(sequences, dtype=np.int32))
    feat_array = tf.convert_to_tensor(np.asarray(feat_array, dtype=np.float32))

    dataset = tf.data.Dataset.from_tensor_slices((feat_array, seq_array))
    if training:
        dataset = dataset.shuffle(buffer_size=max(8, len(seq_array)))

    def _shift(batch_features, sequence):
        decoder_input = sequence[:-1]
        target = sequence[1:]
        return (batch_features, decoder_input), target

    dataset = dataset.map(_shift, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=((tf.TensorShape([49, feat_array.shape[-1]]), tf.TensorShape([None])), tf.TensorShape([None])),
        drop_remainder=False,
    )
    return dataset.prefetch(tf.data.AUTOTUNE)
