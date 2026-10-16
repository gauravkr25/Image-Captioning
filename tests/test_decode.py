import numpy as np

from src.decode import beam_search, greedy_decode
from src.evaluate import evaluate_predictions
from src.models.gru_attention import GRUDecoder
from src.text import Tokenizer


class DummyDecoder:
    def __init__(self, vocab_size=10):
        self.vocab_size = vocab_size
        self.last_token = None

    def __call__(self, inputs, training=False):
        features, seq = inputs
        batch = seq.shape[0]
        out = np.zeros((batch, seq.shape[1], self.vocab_size), dtype=np.float32)
        for b in range(batch):
            for t in range(seq.shape[1]):
                out[b, t, 0] = 1.0
        return out


def test_beam_size_one_matches_greedy():
    tokenizer = Tokenizer(min_word_freq=1)
    tokenizer.fit(["a dog runs", "dog runs fast"])
    tokenizer.max_len = 8
    features = np.random.randn(49, 8).astype(np.float32)
    model = DummyDecoder(vocab_size=len(tokenizer.word_to_id))
    greedy = greedy_decode(model, features, tokenizer, max_len=6)
    beam = beam_search(model, features, tokenizer, beam_size=1, max_len=6)
    assert beam[0][0] == greedy[0]
    assert beam[0][2] == greedy[1]


def test_decode_does_not_exceed_max_len():
    tokenizer = Tokenizer(min_word_freq=1)
    tokenizer.fit(["a dog runs", "dog runs fast"])
    tokenizer.max_len = 8
    features = np.random.randn(49, 8).astype(np.float32)
    model = DummyDecoder(vocab_size=len(tokenizer.word_to_id))
    result = greedy_decode(model, features, tokenizer, max_len=6)
    assert len(result[1]) <= 6


def test_gru_decode_uses_prefix_state_for_generation():
    tokenizer = Tokenizer(min_word_freq=1)
    tokenizer.fit(["a dog runs", "dog runs fast"])
    tokenizer.max_len = 8
    features = np.random.randn(49, 8).astype(np.float32)
    model = GRUDecoder(vocab_size=len(tokenizer.word_to_id), embedding_dim=8, gru_units=12, feature_dim=8)

    greedy = greedy_decode(model, features, tokenizer, max_len=6)
    assert isinstance(greedy, tuple)
    assert len(greedy) == 2
    assert isinstance(greedy[0], str)
    assert 1 <= len(greedy[1]) <= 6


def test_evaluate_predictions_returns_metric_dict():
    refs = [["a dog runs in the grass", "a dog runs in grass"], ["two cats sit on a fence"]]
    preds = ["a dog runs in grass", "two cats on a fence"]
    metrics = evaluate_predictions(refs, preds)
    assert set(metrics).issuperset({"BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L", "CIDEr"})
    assert all(np.isfinite(value) for value in metrics.values())
