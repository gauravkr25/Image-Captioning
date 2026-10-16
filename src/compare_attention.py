
from pathlib import Path
from itertools import combinations

import numpy as np
from PIL import Image

from src.inference import load_bundle, caption_image


RUN_NAME = "flickr8k_gru_attn"
IMAGE_PATH = Path("data/Images/1107246521_d16a476380.jpg")

WORDS_TO_COMPARE = ["dog", "is", "standing"]


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator < 1e-12:
        return float("nan")

    return float(np.dot(a, b) / denominator)


def find_word_maps(result, tokenizer):
    word_to_id = tokenizer.word_to_id
    token_ids = result["attention_token_ids"]
    maps = result["token_attention"]

    found = {}

    for word in WORDS_TO_COMPARE:
        if word not in word_to_id:
            print(f"WARNING: '{word}' is not in the vocabulary.")
            continue

        target_id = int(word_to_id[word])

        matching_maps = [
            np.asarray(attention, dtype=np.float64)
            for token_id, attention in zip(token_ids, maps)
            if int(token_id) == target_id
        ]

        if not matching_maps:
            print(f"WARNING: No attention map found for '{word}'.")
            continue

        # Use the first occurrence if a word appears more than once.
        attention = matching_maps[0]

        if attention.shape != (7, 7):
            print(
                f"WARNING: '{word}' has unexpected shape "
                f"{attention.shape}."
            )
            continue

        found[word] = attention

    return found


def print_map_details(word, attention):
    total = float(attention.sum())
    max_index = int(np.argmax(attention))
    row, col = divmod(max_index, 7)

    print(f"\nWord: {word}")
    print(f"Attention sum: {total:.6f}")
    print(f"Maximum attention: {attention.max():.6f}")
    print(f"Strongest grid cell: row {row + 1}, column {col + 1}")
    print("7x7 attention map:")

    print(
        np.array2string(
            attention,
            precision=3,
            suppress_small=True,
            max_line_width=120,
        )
    )


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH.resolve()}"
        )

    print("Loading model...")
    bundle = load_bundle(RUN_NAME)

    image = Image.open(IMAGE_PATH).convert("RGB")

    print("Generating raw attention maps...")
    result = caption_image(
        bundle,
        image,
        beam_size=3,
        return_attention=True,
    )

    print("\nGenerated caption:", result["caption"])

    found = find_word_maps(result, bundle["tokenizer"])

    if len(found) < 2:
        raise RuntimeError(
            "Could not find enough word-level attention maps. "
            "Check the tokenizer vocabulary and generated caption."
        )

    print("\n" + "=" * 65)
    print("RAW ATTENTION MAP ANALYSIS")
    print("=" * 65)

    for word, attention in found.items():
        print_map_details(word, attention)

    print("\n" + "=" * 65)
    print("PAIRWISE COMPARISON")
    print("=" * 65)

    for word_a, word_b in combinations(found.keys(), 2):
        a = found[word_a]
        b = found[word_b]

        cosine = cosine_similarity(a, b)
        mae = float(np.mean(np.abs(a - b)))
        max_difference = float(np.max(np.abs(a - b)))

        print(f"\n{word_a} vs {word_b}")
        print(f"Cosine similarity: {cosine:.6f}")
        print(f"Mean absolute difference: {mae:.6f}")
        print(f"Maximum absolute difference: {max_difference:.6f}")

    print("\n" + "=" * 65)
    print("Analysis complete.")
    print("Similarity alone does not prove whether attention is correct.")
    print("=" * 65)


if __name__ == "__main__":
    main()
