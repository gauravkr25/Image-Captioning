
from __future__ import annotations

import numpy as np


def _as_logits_array(logits):
    """Convert model output to a one-dimensional vocabulary logits array."""
    if hasattr(logits, "numpy"):
        logits = logits.numpy()

    logits = np.asarray(logits, dtype=np.float32)
    return logits.reshape(-1)


def _state_after_prefix(model, features, seq):
    """Rebuild decoder state for a given token prefix."""
    if not hasattr(model, "decode_step") or not seq:
        return None

    state = None

    for token in seq:
        _, state, _ = model.decode_step(
            np.asarray([token], dtype=np.int32),
            state,
            features,
        )

    return state


def _logits_for_sequence(model, features, seq):
    """Get next-token logits for the current sequence."""
    if hasattr(model, "decode_step"):
        token = seq[-1]
        state = _state_after_prefix(model, features, seq[:-1])

        logits, _, _ = model.decode_step(
            np.asarray([token], dtype=np.int32),
            state,
            features,
        )

        return _as_logits_array(logits)

    logits = model(
        (
            features[None, :, :],
            np.asarray([seq], dtype=np.int32),
        ),
        training=False,
    )

    if hasattr(logits, "numpy"):
        logits = logits.numpy()

    logits = np.asarray(logits, dtype=np.float32)
    return _as_logits_array(logits[0, -1])


def _repeats_ngram(seq, next_token, n=3):
    """
    Check whether adding next_token repeats an existing n-gram.
    The <start> token is excluded from repetition checks.
    """
    tokens = seq[1:]

    if len(tokens) < n - 1:
        return False

    candidate = tuple(tokens[-(n - 1):] + [next_token])

    for i in range(len(tokens) - n + 1):
        existing = tuple(tokens[i:i + n])

        if existing == candidate:
            return True

    return False


def greedy_decode(model, features, tokenizer, max_len):
    """Generate a caption using greedy decoding."""
    features = np.asarray(features, dtype=np.float32)

    start = tokenizer.word_to_id["<start>"]
    end = tokenizer.word_to_id["<end>"]

    seq = [start]

    for _ in range(max_len - 1):
        logits = _logits_for_sequence(model, features, seq)
        token = int(np.argmax(logits))

        seq.append(token)

        if token == end:
            break

    return tokenizer.decode(seq), seq


def beam_search(
    model,
    features,
    tokenizer,
    beam_size=3,
    length_penalty=0.7,
    max_len=20,
    block_ngram_repeat=3,
):
    """
    Generate captions with beam search and repeated n-gram blocking.

    block_ngram_repeat=3 prevents repeating the same trigram.
    Set block_ngram_repeat=0 to disable repetition blocking.
    """
    features = np.asarray(features, dtype=np.float32)

    start = tokenizer.word_to_id["<start>"]
    end = tokenizer.word_to_id["<end>"]

    beams = [([start], 0.0)]
    completed = []

    for _ in range(max_len - 1):
        candidates = []

        for seq, score in beams:
            if seq[-1] == end:
                completed.append((seq, score))
                continue

            logits = _logits_for_sequence(model, features, seq)

            # Convert logits to log probabilities.
            shifted = logits - np.max(logits)
            log_probs = shifted - np.log(
                np.sum(np.exp(shifted)) + 1e-12
            )

            top_indices = np.argsort(log_probs)[::-1][:beam_size]

            for token_idx in top_indices:
                token = int(token_idx)

                # Avoid repeating an existing n-gram.
                if (
                    block_ngram_repeat > 0
                    and _repeats_ngram(
                        seq,
                        token,
                        n=block_ngram_repeat,
                    )
                ):
                    continue

                next_score = score + float(log_probs[token])
                candidates.append((seq + [token], next_score))

        # Fallback if every candidate was blocked.
        if not candidates:
            for seq, score in beams:
                if seq[-1] == end:
                    completed.append((seq, score))
                    continue

                logits = _logits_for_sequence(model, features, seq)
                shifted = logits - np.max(logits)
                log_probs = shifted - np.log(
                    np.sum(np.exp(shifted)) + 1e-12
                )

                # Select the best token that does not repeat an n-gram.
                token = None

                for token_idx in np.argsort(log_probs)[::-1]:
                    candidate_token = int(token_idx)

                    if candidate_token == start:
                        continue

                    if (
                        block_ngram_repeat > 0
                        and _repeats_ngram(
                            seq,
                            candidate_token,
                            n=block_ngram_repeat,
                        )
                    ):
                        continue

                    token = candidate_token
                    break

                # If no unblocked token exists, allow the highest-scoring one.
                if token is None:
                    token = int(np.argmax(log_probs))

                candidates.append(
                    (
                        seq + [token],
                        score + float(log_probs[token]),
                    )
                )

        if not candidates:
            break

        candidates.sort(key=lambda item: item[1], reverse=True)
        beams = candidates[:beam_size]

        if all(seq[-1] == end for seq, _ in beams):
            completed.extend(beams)
            break

    completed.extend(beams)

    # Rank completed sequences using length-normalized log probability.
    ranked = []

    for seq, score in completed:
        generated_length = max(1, len(seq) - 1)

        normalized_score = score / (
            generated_length ** length_penalty
        )

        caption = tokenizer.decode(seq)
        ranked.append((caption, normalized_score, seq))

    # Remove duplicate token sequences.
    unique = {}

    for caption, score, seq in ranked:
        key = tuple(seq)

        if key not in unique or score > unique[key][1]:
            unique[key] = (caption, score, seq)

    ranked = list(unique.values())
    ranked.sort(key=lambda item: item[1], reverse=True)

    return ranked[:beam_size]


def greedy_decode_with_attention(model, features, tokenizer, max_len):
    """
    Generate a greedy caption with placeholder attention maps.

    These maps are placeholders, not actual model attention weights.
    """
    caption, seq = greedy_decode(
        model,
        features,
        tokenizer,
        max_len,
    )

    attention = [
        np.ones((7, 7), dtype=np.float32)
        for _ in seq
    ]

    return caption, seq, attention
