 
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, UnidentifiedImageError

from src.dataset import build_caption_table, split_image_captions


def _make_synthetic_images(root: Path, count: int = 6):
    image_dir = root / "Images"
    image_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(count):
        image = Image.new(
            "RGB",
            (224, 224),
            color=(20 * (idx + 1), 90 + idx * 15, 200 - idx * 10),
        )
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            (30 + idx * 12, 35 + idx * 10, 180 - idx * 9, 180 - idx * 8),
            fill=(80 + idx * 15, 120 + idx * 5, 40 + idx * 20),
        )
        draw.ellipse(
            (110 + idx * 5, 40 + idx * 7, 180 + idx * 5, 120 + idx * 7),
            fill=(220, 190, 110),
        )
        image.save(image_dir / f"synthetic_{idx:02d}.png")

    return image_dir


def _load_image(path: Path, image_size: int):
    try:
        with Image.open(path) as image:
            image = image.convert("RGB").resize((image_size, image_size))
            return np.asarray(image, dtype=np.uint8)
    except (OSError, ValueError) as exc:
        print(f"\n[WARN] Could not read {path.name}: {exc}", flush=True)
        return None


def extract_features(cfg, split: str, limit: int | None = None):
    cfg = dict(cfg)
    data_cfg = cfg.get("data", {})
    root = Path(data_cfg.get("root", "."))
    image_dir = root / "Images"
    image_size = int(data_cfg.get("image_size", 224))

    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")

    if not image_dir.is_dir():
        if not bool(data_cfg.get("smoke") or cfg.get("smoke")):
            raise FileNotFoundError(f"Images directory not found: {image_dir}")
        image_dir = _make_synthetic_images(root)

    selected_names = None
    for filename in ("captions.txt", "Flickr8k.token.txt", "flickr8k.token.txt"):
        if (root / filename).exists():
            table = build_caption_table(root)
            splits = split_image_captions(
                table,
                val_fraction=float(data_cfg.get("val_fraction", 0.1)),
                test_fraction=float(data_cfg.get("test_fraction", 0.1)),
                seed=int(cfg.get("seed", 42)),
            )
            selected_names = {
                str(name).strip()
                for name in splits[split]["image"].dropna()
            }
            break

    images = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if selected_names is not None:
        images = [path for path in images if path.name in selected_names]

    if not images:
        raise RuntimeError(
            f"No images found for split '{split}' in {image_dir}. "
            "Check the dataset and image filenames."
        )

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
        images = images[:limit]

    backbone_name = cfg.get("encoder", {}).get("backbone", "EfficientNetV2B0")
    backbone_class = getattr(tf.keras.applications, backbone_name, None)
    if backbone_class is None:
        raise ValueError(f"Unsupported Keras application backbone: {backbone_name}")

    print(f"Loading {backbone_name}...", flush=True)
    backbone = backbone_class(include_top=False, weights="imagenet")
    backbone.trainable = False

    configured_batch = int(cfg.get("encoder", {}).get("batch_size", 16))
    batch_size = max(1, min(configured_batch, 16))
    print(
        f"Split: {split} | Images: {len(images)} | "
        f"Batch size: {batch_size} | Image size: {image_size}",
        flush=True,
    )

    output_dir = Path("artifacts/features") / backbone_name
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / f"{split}.npy"
    index_path = output_dir / f"{split}_index.json"

    # Write temporary files first; preserve previous outputs until success.
    temp_path = output_dir / f"{split}.tmp.npy"
    temp_index_path = output_dir / f"{split}_index.tmp.json"

    feature_chunks = []
    index = {}
    failed = 0
    processed = 0

    try:
        for start in range(0, len(images), batch_size):
            batch_paths = images[start:start + batch_size]
            batch_images = []
            batch_names = []

            for image_path in batch_paths:
                arr = _load_image(image_path, image_size)
                if arr is None:
                    failed += 1
                    continue
                batch_images.append(arr)
                batch_names.append(image_path.name)

            if not batch_images:
                continue

            try:
                tensor = tf.convert_to_tensor(np.stack(batch_images))
                tensor = tf.keras.applications.efficientnet_v2.preprocess_input(
                    tensor
                )
                output = backbone(tensor, training=False)
                output = tf.reshape(output, (len(batch_images), -1, output.shape[-1]))
                batch_features = output.numpy().astype(np.float16)

            except Exception as batch_error:
                print(
                    f"\n[WARN] Batch failed at image {start + 1}: "
                    f"{batch_error}. Retrying images individually.",
                    flush=True,
                )
                batch_features = []
                successful_names = []

                for image_array, image_name in zip(batch_images, batch_names):
                    try:
                        tensor = tf.convert_to_tensor(image_array[None, ...])
                        tensor = tf.keras.applications.efficientnet_v2.preprocess_input(
                            tensor
                        )
                        output = backbone(tensor, training=False)
                        output = tf.reshape(output, (1, -1, output.shape[-1]))
                        batch_features.append(output.numpy()[0].astype(np.float16))
                        successful_names.append(image_name)
                    except Exception as image_error:
                        failed += 1
                        print(
                            f"\n[ERROR] {image_name}: {image_error}",
                            flush=True,
                        )

                batch_names = successful_names
                if not batch_features:
                    continue
                batch_features = np.stack(batch_features)

            for name, feature in zip(batch_names, batch_features):
                index[name] = processed
                processed += 1
                feature_chunks.append(feature[None, ...])

            print(
                f"\rProgress: {min(start + len(batch_paths), len(images))}/"
                f"{len(images)} | Successful: {processed} | Failed: {failed}",
                end="",
                flush=True,
            )

        print("", flush=True)

        if processed == 0:
            raise RuntimeError("Feature extraction produced zero valid features.")

        all_features = np.concatenate(feature_chunks, axis=0)

        with open(temp_path, "wb") as handle:
            np.save(handle, all_features, allow_pickle=False)

        with temp_index_path.open("w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2)

        # Publish only after extraction and both temporary writes succeed.
        os.replace(temp_path, save_path)
        os.replace(temp_index_path, index_path)

        print(f"Saved features: {save_path}", flush=True)
        print(f"Saved index: {index_path}", flush=True)
        print(f"Final shape: {all_features.shape}", flush=True)
        print(f"Successful: {processed} | Failed: {failed}", flush=True)

        return {"features": all_features, "index": index}

    finally:
        for temporary_file in (temp_path, temp_index_path):
            if temporary_file.exists():
                temporary_file.unlink()


if __name__ == "__main__":
    import argparse
    from src.config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N selected images for a quick test.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    extract_features(config, args.split, limit=args.limit)
