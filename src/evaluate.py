
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge

from src.config import load_config
from src.dataset import build_caption_table, split_image_captions
from src.decode import beam_search, greedy_decode
from src.features import extract_features
from src.inference import load_bundle


def evaluate_predictions(references, predictions):
    """Calculate corpus-level BLEU, ROUGE-L, and CIDEr metrics."""

    if not references or not predictions:
        raise ValueError(
            "No references or predictions available for evaluation."
        )

    if len(references) != len(predictions):
        raise ValueError(
            "Reference and prediction counts do not match."
        )

    ref = {
        str(i): [str(caption) for caption in refs]
        for i, refs in enumerate(references)
    }

    hyp = {
        str(i): [str(prediction)]
        for i, prediction in enumerate(predictions)
    }

    bleu_scores, _ = Bleu(4).compute_score(ref, hyp)
    cider_score, _ = Cider().compute_score(ref, hyp)
    rouge_score, _ = Rouge().compute_score(ref, hyp)

    return {
        "BLEU-1": float(bleu_scores[0]),
        "BLEU-2": float(bleu_scores[1]),
        "BLEU-3": float(bleu_scores[2]),
        "BLEU-4": float(bleu_scores[3]),
        "ROUGE-L": float(rouge_score),
        "CIDEr": float(cider_score),
    }


def _synthetic_eval_rows(seed=7, n=8):
    """Create synthetic rows for development only."""

    captions = [
        "a dog runs in the grass",
        "two cats stare at the sky",
        "a person holds a ball",
        "a red train in a station",
        "a small puppy jumps in water",
        "a dog sits near a tree",
    ]

    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, 49, 1280)).astype(np.float32)
    references = []

    for i in range(n):
        base = captions[i % len(captions)]

        variants = [
            base,
            base.replace("dog", "puppy") if "dog" in base else base,
            base + " outside",
            "a photo of " + base,
        ]

        references.append(variants[:3])

    image_names = [
        f"synthetic_{i:04d}.jpg"
        for i in range(n)
    ]

    return features, references, image_names


def _load_real_eval_rows(cfg, split):
    """Load real Flickr8k features, references, and image filenames."""

    root = Path(cfg.get("data", {}).get("root", "."))

    if not root.exists():
        return None

    caption_files = [
        "captions.txt",
        "Flickr8k.token.txt",
        "flickr8k.token.txt",
    ]

    if not any((root / filename).exists() for filename in caption_files):
        return None

    caption_table = build_caption_table(root)

    split_table = split_image_captions(
        caption_table,
        val_fraction=float(
            cfg.get("data", {}).get("val_fraction", 0.1)
        ),
        test_fraction=float(
            cfg.get("data", {}).get("test_fraction", 0.1)
        ),
        seed=int(cfg.get("seed", 42)),
    )

    if split not in split_table:
        raise ValueError(f"Unknown split: {split}")

    split_rows = split_table[split]

    image_names = sorted(
        str(name)
        for name in split_rows["image"].dropna().unique()
    )

    if not image_names:
        return None

    backbone = cfg.get("encoder", {}).get(
        "backbone",
        "EfficientNetV2B0",
    )

    feature_path = (
        Path("artifacts/features") / backbone / f"{split}.npy"
    )

    index_path = (
        Path("artifacts/features") / backbone / f"{split}_index.json"
    )

    if not feature_path.exists() or not index_path.exists():
        feature_bundle = extract_features(cfg, split)
        features = feature_bundle["features"]
        index = feature_bundle["index"]
    else:
        features = np.load(feature_path, mmap_mode="r")

        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)

    image_to_captions = {}

    for row in split_rows.to_dict(orient="records"):
        image_name = str(row["image"]).strip()
        caption = str(row["caption"]).strip()

        if image_name and caption:
            image_to_captions.setdefault(
                image_name, []
            ).append(caption)

    ordered_features = []
    references = []
    ordered_image_names = []

    # Build all three lists together to preserve exact alignment.
    for image_name in image_names:
        if image_name not in index:
            continue

        captions = image_to_captions.get(image_name, [])

        if not captions:
            continue

        feature_index = index[image_name]

        ordered_features.append(features[feature_index])
        references.append(captions)
        ordered_image_names.append(image_name)

    if not ordered_features:
        return None

    return (
        np.stack(ordered_features).astype(np.float32),
        references,
        ordered_image_names,
    )


def _decode_predictions(
    bundle,
    features,
    tokenizer,
    beam_size=3,
    max_len=12,
):
    """Generate captions using greedy decoding or beam search."""

    decoder = bundle["decoder"]

    if decoder is None:
        raise ValueError("No decoder loaded for this run.")

    predictions = []
    sample_details = []

    total = len(features)

    for i, feature in enumerate(features, start=1):
        if beam_size == 1:
            caption, token_ids = greedy_decode(
                decoder,
                feature,
                tokenizer,
                max_len=max_len,
            )
        else:
            candidates = beam_search(
                decoder,
                feature,
                tokenizer,
                beam_size=beam_size,
                max_len=max_len,
            )

            if not candidates:
                caption, token_ids = greedy_decode(
                    decoder,
                    feature,
                    tokenizer,
                    max_len=max_len,
                )
            else:
                caption, _, token_ids = candidates[0]

        predictions.append(caption)

        sample_details.append({
            "sample": i,
            "prediction": caption,
            "token_ids": [int(token) for token in token_ids],
        })

        print(
            f"\rGenerating captions: {i}/{total}",
            end="",
            flush=True,
        )

    if total:
        print()

    return predictions, sample_details


def _summarize_samples(
    references,
    predictions,
    image_names=None,
):
    """Calculate per-image CIDEr scores and preserve filenames."""

    if image_names is None:
        image_names = [
            f"image_{i:04d}"
            for i in range(len(predictions))
        ]

    if not (
        len(references) == len(predictions) == len(image_names)
    ):
        raise ValueError(
            "References, predictions, and image names must have "
            "the same length."
        )

    rows = []

    for image_name, ref_list, prediction in zip(
        image_names,
        references,
        predictions,
    ):
        try:
            scores = evaluate_predictions(
                [ref_list],
                [prediction],
            )
            cider_score = scores["CIDEr"]
        except Exception:
            cider_score = 0.0

        rows.append({
            "image_id": image_name,
            "references": ref_list,
            "prediction": prediction,
            "CIDEr": cider_score,
        })

    rows.sort(
        key=lambda row: row["CIDEr"],
        reverse=True,
    )

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an image-captioning model."
    )

    parser.add_argument("--run", default="smoke")

    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max-len",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only the first N images; 0 means all images.",
    )

    args = parser.parse_args()

    if args.beam_size < 1:
        parser.error("--beam-size must be at least 1.")

    if args.max_len < 2:
        parser.error("--max-len must be at least 2.")

    if args.limit < 0:
        parser.error("--limit cannot be negative.")

    report_dir = Path("artifacts/reports") / args.run
    report_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(args.run)
    tokenizer = bundle["tokenizer"]

    if not hasattr(tokenizer, "word_to_id"):
        raise ValueError(
            "Loaded tokenizer does not expose word_to_id."
        )

    config_path = (
        Path("artifacts/models") / args.run / "config.yaml"
    )

    cfg = (
        load_config(config_path)
        if config_path.exists()
        else {}
    )

    if args.run == "smoke":
        real_eval_rows = None
    else:
        real_eval_rows = _load_real_eval_rows(
            cfg,
            args.split,
        )

    if real_eval_rows is None:
        print(
            "WARNING: Real evaluation data could not be loaded. "
            "Using synthetic data; these scores are NOT valid "
            "for reporting model performance."
        )

        features, references, image_names = _synthetic_eval_rows()
        evaluation_source = "synthetic"
    else:
        features, references, image_names = real_eval_rows
        evaluation_source = "Flickr8k"

    # Apply the same limit to features, references, and filenames.
    if args.limit:
        features = features[:args.limit]
        references = references[:args.limit]
        image_names = image_names[:args.limit]

    if len(features) == 0:
        raise ValueError("No images available to evaluate.")

    if not (
        len(features) == len(references) == len(image_names)
    ):
        raise ValueError(
            "Feature, reference, and image filename counts do not match."
        )

    print(f"Evaluation source: {evaluation_source}")
    print(f"Split: {args.split}")
    print(f"Images: {len(features)}")
    print(f"Beam size: {args.beam_size}")
    print(f"Maximum caption length: {args.max_len}")

    predictions, sample_details = _decode_predictions(
        bundle,
        features,
        tokenizer,
        beam_size=args.beam_size,
        max_len=args.max_len,
    )

    metrics = evaluate_predictions(
        references,
        predictions,
    )

    metrics.update({
        "run": args.run,
        "split": args.split,
        "num_samples": len(predictions),
        "beam_size": args.beam_size,
        "max_len": args.max_len,
        "evaluation_source": evaluation_source,
    })

    sample_rows = _summarize_samples(
        references,
        predictions,
        image_names,
    )

    report = {
        "metrics": metrics,
        "samples": sample_rows,
        "best": sample_rows[:min(10, len(sample_rows))],
        "median": sample_rows[
            max(0, len(sample_rows) // 2 - 5):
            len(sample_rows) // 2 + 5
        ],
        "worst": list(
            reversed(
                sample_rows[-min(10, len(sample_rows)):]
            )
        ),
        "generated": sample_details,
    }

    metrics_path = report_dir / "metrics.json"
    samples_path = report_dir / "samples.json"

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    with samples_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\nEvaluation metrics:")
    print(json.dumps(metrics, indent=2))

    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Sample report saved to: {samples_path}")


if __name__ == "__main__":
    main()
