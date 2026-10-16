
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.inference import load_bundle, caption_image


RUN_NAME = "flickr8k_gru_attn"
IMAGE_PATH = Path("data/Images/1107246521_d16a476380.jpg")
OUTPUT_DIR = Path("outputs/attention_aligned")

SPECIAL_TOKENS = {"<pad>", "<start>", "<end>", "<unk>"}


def get_word(tokenizer, token_id):
    """Convert a token ID into its vocabulary word."""
    if hasattr(tokenizer, "id_to_word"):
        vocab = tokenizer.id_to_word
        if isinstance(vocab, dict):
            return vocab.get(int(token_id), "<unk>")
        if isinstance(vocab, (list, tuple)):
            idx = int(token_id)
            return vocab[idx] if 0 <= idx < len(vocab) else "<unk>"

    if hasattr(tokenizer, "word_to_id"):
        reverse_vocab = {
            int(idx): word
            for word, idx in tokenizer.word_to_id.items()
        }
        return reverse_vocab.get(int(token_id), "<unk>")

    raise AttributeError(
        "Cannot find token vocabulary in Tokenizer. "
        "Check src/text.py."
    )


def save_attention_image(image, attention, word, index):
    attention = np.asarray(attention, dtype=np.float32)

    if attention.shape != (7, 7):
        raise ValueError(
            f"Expected a 7x7 attention map for '{word}', "
            f"received {attention.shape}"
        )

    attention = np.maximum(attention, 0)
    maximum = float(attention.max())

    if maximum > 0:
        attention /= maximum

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(image)
    heatmap = axes[1].imshow(
        attention,
        cmap="jet",
        alpha=0.55,
        interpolation="bilinear",
        extent=(0, image.width, image.height, 0),
    )
    axes[1].set_title(f"Attention for: {word}")
    axes[1].axis("off")

    fig.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()

    safe_word = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in word
    )
    output_path = OUTPUT_DIR / f"word_{index:02d}_{safe_word}.png"

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH.resolve()}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    bundle = load_bundle(RUN_NAME)

    print(f"Loading image: {IMAGE_PATH}")
    image = Image.open(IMAGE_PATH).convert("RGB")

    print("Generating caption and attention maps...")

    # IMPORTANT: caption_image expects (bundle, PIL image).
    result = caption_image(
        bundle,
        image,
        beam_size=3,
        return_attention=True,
    )

    caption = result["caption"]
    token_ids = result["attention_token_ids"]
    attention_maps = result["token_attention"]

    print("\n" + "=" * 50)
    print("Generated caption:", caption)
    print("Token IDs:", token_ids)
    print("Attention maps:", len(attention_maps))
    print("=" * 50)

    if len(token_ids) != len(attention_maps):
        raise ValueError(
            f"Mismatch: {len(token_ids)} token IDs and "
            f"{len(attention_maps)} attention maps."
        )

    visible_words = []
    saved_count = 0

    for token_id, attention in zip(token_ids, attention_maps):
        word = get_word(bundle["tokenizer"], token_id)

        if word in SPECIAL_TOKENS:
            continue

        visible_words.append(word)
        saved_count += 1

        path = save_attention_image(
            image=image,
            attention=attention,
            word=word,
            index=saved_count,
        )
        print("Saved:", path)

    reconstructed_caption = " ".join(visible_words)

    print("\nCaption reconstructed from token IDs:")
    print(reconstructed_caption)
    print("Images saved:", saved_count)
    print("Output folder:", OUTPUT_DIR.resolve())

    if reconstructed_caption == caption:
        print("SUCCESS: Caption words and attention maps are aligned.")
    else:
        print(
            "WARNING: Caption text differs from the visible token IDs. "
            "Check tokenizer mapping before interpreting the heatmaps."
        )


if __name__ == "__main__":
    main()
