from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

SPECIAL_TOKENS = ["<pad>", "<start>", "<end>", "<unk>"]


def clean_caption(s: str) -> str:
    text = str(s or "").lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [token for token in text.split() if not re.search(r"\d", token)]
    return " ".join(tokens)


class Tokenizer:
    def __init__(self, min_word_freq: int = 1):
        self.min_word_freq = min_word_freq
        self.word_to_id = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
        self.id_to_word = {idx: token for token, idx in self.word_to_id.items()}
        self._max_len = None

    @property
    def max_len(self):
        return self._max_len

    @max_len.setter
    def max_len(self, value: int):
        self._max_len = int(value)

    def fit(self, captions):
        counts: Counter[str] = Counter()
        for caption in captions:
            cleaned = clean_caption(caption)
            if not cleaned:
                continue
            for word in cleaned.split():
                counts[word] += 1

        vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
        for word, count in sorted(counts.items()):
            if count >= self.min_word_freq:
                vocab[word] = len(vocab)
        self.word_to_id = vocab
        self.id_to_word = {idx: token for token, idx in vocab.items()}
        return len(vocab)

    def encode(self, text: str, max_len: int | None = None):
        if max_len is None:
            max_len = self._max_len or 16
        max_len = max(1, int(max_len))
        cleaned = clean_caption(text)
        tokens = [self.word_to_id.get(word, self.word_to_id["<unk>"]) for word in cleaned.split()]
        encoded = [self.word_to_id["<start>"]] + tokens + [self.word_to_id["<end>"]]
        if len(encoded) > max_len:
            encoded = encoded[: max_len - 1] + [self.word_to_id["<end>"]]
        if len(encoded) < max_len:
            encoded = encoded + [self.word_to_id["<pad>"]] * (max_len - len(encoded))
        return encoded[:max_len]

    def decode(self, ids):
        words = []
        for token_id in ids:
            token = self.id_to_word.get(int(token_id))
            if token is None:
                continue
            if token == "<pad>":
                continue
            if token == "<start>":
                continue
            if token == "<end>":
                break
            if token == "<unk>":
                continue
            words.append(token)
        return " ".join(words)

    def save(self, path):
        payload = {
            "min_word_freq": self.min_word_freq,
            "word_to_id": self.word_to_id,
            "id_to_word": self.id_to_word,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        tokenizer = cls(min_word_freq=payload.get("min_word_freq", 1))
        tokenizer.word_to_id = payload["word_to_id"]
        tokenizer.id_to_word = {int(k): v for k, v in payload["id_to_word"].items()}
        return tokenizer
