from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.config import load_config, save_config
from src.dataset import build_caption_table, split_image_captions
from src.models.common import WarmupCosine, masked_sparse_ce
from src.models.gru_attention import GRUDecoder
from src.models.transformer import CaptionTransformer
from src.text import Tokenizer


def build_tokenizer(cfg, captions):
    tokenizer = Tokenizer(
        min_word_freq=int(
            cfg.get("data", {}).get("min_word_freq", 1)
        )
    )
    tokenizer.fit(captions)
    tokenizer.max_len = int(
        cfg.get("data", {}).get("max_len", 22)
    )
    return tokenizer


def build_model(cfg, vocab_size):
    model_cfg = cfg.get("model", {})

    if model_cfg.get("type", "transformer") == "gru_attention":
        return GRUDecoder(
            vocab_size=vocab_size,
            embedding_dim=int(model_cfg.get("embed_dim", 256)),
            gru_units=int(model_cfg.get("gru_units", 512)),
            feature_dim=1280,
            dropout=float(model_cfg.get("dropout", 0.1)),
        )

    return CaptionTransformer(
        vocab_size=vocab_size,
        d_model=int(model_cfg.get("d_model", 128)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        dff=int(model_cfg.get("dff", 512)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )


def _get_splits(cfg):
    root = Path(cfg["data"]["root"])
    table = build_caption_table(root)

    return split_image_captions(
        table,
        val_fraction=float(
            cfg["data"].get("val_fraction", 0.1)
        ),
        test_fraction=float(
            cfg["data"].get("test_fraction", 0.1)
        ),
        seed=int(cfg.get("seed", 42)),
    )


def _load_real_split(cfg, split):
    """
    Compatibility helper for existing tests.

    Loads cached image features and expands them to one row
    per caption. Use load_split() for memory-efficient training.
    """
    splits = _get_splits(cfg)
    rows = splits[split]

    backbone = cfg.get("encoder", {}).get(
        "backbone", "EfficientNetV2B0"
    )

    feature_path = (
        Path("artifacts")
        / "features"
        / backbone
        / f"{split}.npy"
    )
    index_path = (
        Path("artifacts")
        / "features"
        / backbone
        / f"{split}_index.json"
    )

    if not feature_path.exists() or not index_path.exists():
        raise FileNotFoundError(
            f"Missing cached features for '{split}'. "
            "Run the feature extraction script first."
        )

    features = np.load(feature_path, mmap_mode="r")

    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)

    feature_rows = []
    captions = []

    for row in rows.to_dict(orient="records"):
        image_name = str(row["image"]).strip()
        caption = str(row["caption"]).strip()

        if not image_name or not caption:
            continue

        if image_name not in index:
            continue

        feature_rows.append(
            np.asarray(features[index[image_name]], dtype=np.float32)
        )
        captions.append(caption)

    if not captions:
        raise RuntimeError(
            f"No captions with cached features found for '{split}'."
        )

    return np.stack(feature_rows), captions


def load_split(cfg, split):
    """
    Memory-efficient loader:
    keeps one feature row per image and maps captions to feature indices.
    """
    splits = _get_splits(cfg)
    rows = splits[split]

    backbone = cfg.get("encoder", {}).get(
        "backbone", "EfficientNetV2B0"
    )

    feature_path = (
        Path("artifacts")
        / "features"
        / backbone
        / f"{split}.npy"
    )
    index_path = (
        Path("artifacts")
        / "features"
        / backbone
        / f"{split}_index.json"
    )

    if not feature_path.exists() or not index_path.exists():
        raise FileNotFoundError(
            f"Missing {split} features. "
            "Run src.features for this split first."
        )

    features = np.load(feature_path, mmap_mode="r")

    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)

    grouped = {}

    for row in rows.to_dict(orient="records"):
        image_name = str(row["image"]).strip()
        caption = str(row["caption"]).strip()

        if image_name and caption and image_name in index:
            grouped.setdefault(image_name, []).append(caption)

    image_names = sorted(grouped)

    if not image_names:
        raise RuntimeError(
            f"No images with cached features found for '{split}'."
        )

    captions = []
    feature_indices = []

    for image_name in image_names:
        for caption in grouped[image_name]:
            captions.append(caption)
            feature_indices.append(index[image_name])

    return (
        features,
        image_names,
        grouped,
        captions,
        feature_indices,
    )


def make_caption_dataset(
    features,
    captions,
    feature_indices,
    tokenizer,
    batch_size,
    training,
    max_len,
):
    caption_ids = np.asarray(
        [
            tokenizer.encode(caption, max_len=max_len)
            for caption in captions
        ],
        dtype=np.int32,
    )

    indices = np.asarray(feature_indices, dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (indices, caption_ids)
    )

    if training:
        dataset = dataset.shuffle(
            min(len(indices), 4096),
            seed=42,
            reshuffle_each_iteration=True,
        )

    def load_pair(index, sequence):
        feature = tf.numpy_function(
            lambda i: np.asarray(
                features[int(i)], dtype=np.float32
            ),
            [index],
            tf.float32,
        )
        feature.set_shape([49, 1280])

        decoder_input = sequence[:-1]
        target = sequence[1:]

        return (feature, decoder_input), target

    return (
        dataset
        .map(load_pair, num_parallel_calls=1)
        .batch(batch_size)
        .prefetch(1)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load saved best weights and continue training (optimizer state is reset).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_cfg = dict(cfg)

    data_cfg = dict(run_cfg.get("data", {}))
    train_cfg = dict(run_cfg.get("train", {}))

    if args.smoke or bool(data_cfg.get("smoke", False)):
        raise SystemExit(
            "Smoke mode is not enabled in this trainer yet. "
            "Run pytest before starting full training."
        )

    run_cfg["data"] = data_cfg
    run_cfg["train"] = train_cfg

    train_data = load_split(run_cfg, "train")
    val_data = load_split(run_cfg, "val")

    train_features, _, _, train_captions, train_indices = train_data
    val_features, _, _, val_captions, val_indices = val_data

    tokenizer = build_tokenizer(run_cfg, train_captions)
    vocab_size = len(tokenizer.word_to_id)
    max_len = tokenizer.max_len

    batch_size = int(train_cfg.get("batch_size", 16))
    epochs = int(train_cfg.get("epochs", 40))
    patience = int(
        train_cfg.get("early_stopping_patience", 5)
    )

    train_ds = make_caption_dataset(
        train_features,
        train_captions,
        train_indices,
        tokenizer,
        batch_size,
        True,
        max_len,
    )

    val_ds = make_caption_dataset(
        val_features,
        val_captions,
        val_indices,
        tokenizer,
        batch_size,
        False,
        max_len,
    )

    steps_per_epoch = (
        len(train_captions) + batch_size - 1
    ) // batch_size

    total_steps = max(1, epochs * steps_per_epoch)

    schedule = WarmupCosine(
        peak_lr=float(train_cfg.get("peak_lr", 5e-4)),
        warmup_steps=int(
            train_cfg.get("warmup_steps", 1000)
        ),
        total_steps=total_steps,
    )

    model = build_model(run_cfg, vocab_size)

    run_name = run_cfg.get("run_name", "gru_attention")
    save_dir = Path("artifacts/models") / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build subclassed Keras model before loading weights.
    dummy_features = tf.zeros((1, 49, 1280), dtype=tf.float32)
    dummy_tokens = tf.zeros((1, max(1, max_len - 1)), dtype=tf.int32)
    _ = model((dummy_features, dummy_tokens), training=False)

    best_weights_path = save_dir / "best.weights.h5"
    last_weights_path = save_dir / "last.weights.h5"
    progress_path = save_dir / "progress.json"
    initial_epoch = 0

    if args.resume:
        resume_path = last_weights_path if last_weights_path.exists() else best_weights_path
        if not resume_path.exists():
            raise FileNotFoundError(
                f"Cannot resume: no saved weights found at {last_weights_path} "
                f"or {best_weights_path}."
            )
        print(f"Loading weights from: {resume_path}")
        model.load_weights(str(resume_path))
        if progress_path.exists():
            with progress_path.open("r", encoding="utf-8") as handle:
                progress = json.load(handle)
            initial_epoch = int(progress.get("completed_epochs", 0))
        if initial_epoch >= epochs:
            print(
                f"Saved epoch count ({initial_epoch}) is already at or above "
                f"configured epochs ({epochs}). Set train.epochs higher to continue."
            )
            return
        print(
            f"Continuing from saved weights; next epoch is {initial_epoch + 1}. "
            "Note: optimizer state is reset when resuming from weights-only files."
        )

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=schedule
        ),
        loss=lambda y, p: masked_sparse_ce(
            y,
            p,
            label_smoothing=float(
                train_cfg.get("label_smoothing", 0.1)
            ),
        ),
    )

    save_config(save_dir / "config.yaml", run_cfg)
    tokenizer.save(save_dir / "tokenizer.json")

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(save_dir / "best.weights.h5"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    class SaveLastWeightsAndProgress(tf.keras.callbacks.Callback):
        """Save a recoverable weights checkpoint at every completed epoch."""
        def on_epoch_end(self, epoch, logs=None):
            self.model.save_weights(str(last_weights_path))
            with progress_path.open("w", encoding="utf-8") as handle:
                json.dump({"completed_epochs": int(epoch + 1)}, handle, indent=2)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    history_path = (
        Path("artifacts/reports") / run_name / "history.csv"
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    csv_logger = tf.keras.callbacks.CSVLogger(
        str(history_path), append=args.resume and history_path.exists()
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=initial_epoch,
        epochs=epochs,
        callbacks=[checkpoint, SaveLastWeightsAndProgress(), early_stop, csv_logger],
        verbose=1,
    )

    model.save(save_dir / "decoder.keras")
    model.save_weights(save_dir / "weights.weights.h5")

    print(f"Saved model bundle: {save_dir}")
    print(f"Saved history: {history_path}")


if __name__ == "__main__":
    main()
