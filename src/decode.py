
from __future__ import annotations

import numpy as np


def _as_logits_array(logits):
    if hasattr(logits, "numpy"):
        logits = logits.numpy()
    return np.asarray(logits, dtype=np.float32).reshape(-1)


def _as_attention_map(attention):
    if attention is None:
        return None

    if hasattr(attention, "numpy"):
        attention = attention.numpy()

    attention = np.asarray(attention, dtype=np.float32).reshape(-1)

    if attention.size != 49:
        return None

    return attention.reshape(7, 7)


def _state_after_prefix(model, features, seq):
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


def _logits_and_attention_for_sequence(model, features, seq):
    if hasattr(model, "decode_step"):
        state = _state_after_prefix(model, features, seq[:-1])

        logits, _, attention = model.decode_step(
            np.asarray([seq[-1]], dtype=np.int32),
            state,
            features,
        )

        return _as_logits_array(logits), _as_attention_map(attention)

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
    return _as_logits_array(logits[0, -1]), None


def _logits_for_sequence(model, features, seq):
    logits, _ = _logits_and_attention_for_sequence(model, features, seq)
    return logits


def _repeats_ngram(seq, next_token, n=3):
    tokens = list(seq[1:])

    if n <= 1:
        return next_token in tokens

    if len(tokens) < n - 1:
        return False

    candidate = tuple(tokens[-(n - 1):] + [next_token])

    for i in range(len(tokens) - n + 1):
        if tuple(tokens[i:i + n]) == candidate:
            return True

    return False


def _select_token(logits, seq, end, start, block_ngram_repeat=3):
    logits = np.asarray(logits, dtype=np.float32).reshape(-1).copy()

    if not np.any(np.isfinite(logits)):
        return int(end)

    if 0 <= start < len(logits):
        logits[start] = -np.inf

    for token_id in np.argsort(logits)[::-1]:
        token_id = int(token_id)

        if not np.isfinite(logits[token_id]):
            continue

        if token_id == end:
            return token_id

        if len(seq) > 1 and token_id == seq[-1]:
            continue

        if block_ngram_repeat > 0 and _repeats_ngram(
            seq, token_id, block_ngram_repeat
        ):
            continue

        return token_id

    return int(end)


def greedy_decode(
    model,
    features,
    tokenizer,
    max_len,
    block_ngram_repeat=3,
):
    features = np.asarray(features, dtype=np.float32)
    start = tokenizer.word_to_id["<start>"]
    end = tokenizer.word_to_id["<end>"]
    seq = [start]

    for _ in range(max_len - 1):
        logits = _logits_for_sequence(model, features, seq)
        token = _select_token(
            logits, seq, end, start, block_ngram_repeat
        )
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
    return_attention=False,
):
    """
    Return:
      Normal: (caption, score, token_ids)
      With attention: (caption, score, token_ids, token_attention)

    token_attention[i] is the map used to predict token_ids[i + 1].
    Therefore token_attention[0] corresponds to the first generated
    token after <start>, including <end> if it is generated.
    """
    features = np.asarray(features, dtype=np.float32)
    start = tokenizer.word_to_id["<start>"]
    end = tokenizer.word_to_id["<end>"]

    # Each beam stores (sequence, score, token-attention pairs).
    beams = [([start], 0.0, [])]
    completed = []

    for _ in range(max_len - 1):
        candidates = []

        for seq, score, history in beams:
            if seq[-1] == end:
                completed.append((seq, score, history))
                continue

            logits, attention_map = _logits_and_attention_for_sequence(
                model, features, seq
            )

            if not np.any(np.isfinite(logits)):
                token = end
                new_history = list(history)
                if attention_map is not None:
                    new_history.append((token, attention_map.copy()))
                candidates.append(
                    (seq + [token], score, new_history)
                )
                continue

            shifted = logits - np.max(logits)
            log_probs = shifted - np.log(
                np.sum(np.exp(shifted)) + 1e-12
            )

            top_count = min(
                len(log_probs),
                max(beam_size * 5, beam_size),
            )

            for token_idx in np.argsort(log_probs)[::-1][:top_count]:
                token = int(token_idx)

                if token == start or not np.isfinite(log_probs[token]):
                    continue

                if token != end:
                    if len(seq) > 1 and token == seq[-1]:
                        continue

                    if block_ngram_repeat > 0 and _repeats_ngram(
                        seq, token, block_ngram_repeat
                    ):
                        continue

                new_history = list(history)

                if attention_map is not None:
                    new_history.append((token, attention_map.copy()))

                candidates.append(
                    (
                        seq + [token],
                        score + float(log_probs[token]),
                        new_history,
                    )
                )

        if not candidates:
            for seq, score, history in beams:
                if seq[-1] != end:
                    seq = seq + [end]
                completed.append((seq, score, history))
            break

        candidates.sort(key=lambda item: item[1], reverse=True)
        beams = candidates[:beam_size]

        if all(seq[-1] == end for seq, _, _ in beams):
            completed.extend(beams)
            break

    completed.extend(beams)

    ranked = []

    for seq, score, history in completed:
        generated_length = max(1, len(seq) - 1)
        normalized_score = score / (
            generated_length ** length_penalty
        )
        caption = tokenizer.decode(seq)

        if return_attention:
            # Store token IDs alongside maps so callers can align them.
            ranked.append(
                (caption, normalized_score, seq, history)
            )
        else:
            ranked.append((caption, normalized_score, seq))

    unique = {}

    for item in ranked:
        key = tuple(item[2])
        if key not in unique or item[1] > unique[key][1]:
            unique[key] = item

    ranked = list(unique.values())
    ranked.sort(key=lambda item: item[1], reverse=True)

    return ranked[:beam_size]


def greedy_decode_with_attention(model, features, tokenizer, max_len):
    features = np.asarray(features, dtype=np.float32)
    start = tokenizer.word_to_id["<start>"]
    end = tokenizer.word_to_id["<end>"]

    seq = [start]
    attention_history = []

    for _ in range(max_len - 1):
        logits, attention_map = _logits_and_attention_for_sequence(
            model, features, seq
        )

        token = _select_token(
            logits, seq, end, start, block_ngram_repeat=3
        )

        if attention_map is not None:
            attention_history.append(
                (token, attention_map.copy())
            )

        seq.append(token)

        if token == end:
            break

    return tokenizer.decode(seq), seq, attention_history
