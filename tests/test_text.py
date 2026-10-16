from src.text import Tokenizer, clean_caption


def test_clean_caption_removes_digits_and_punctuation():
    assert clean_caption("A dog!!! runs 123 at 5pm") == "a dog runs at pm"


def test_tokenizer_round_trip():
    tokenizer = Tokenizer(min_word_freq=1)
    tokenizer.fit(["a dog runs", "dog runs fast"])
    encoded = tokenizer.encode("a dog runs", max_len=8)
    assert encoded[0] == tokenizer.word_to_id["<start>"]
    assert encoded[-1] == tokenizer.word_to_id["<pad>"]
    decoded = tokenizer.decode(encoded)
    assert "dog" in decoded
    assert "runs" in decoded


def test_special_tokens_are_fixed():
    tokenizer = Tokenizer(min_word_freq=1)
    tokenizer.fit(["a dog"])
    assert tokenizer.word_to_id["<pad>"] == 0
    assert tokenizer.word_to_id["<start>"] == 1
    assert tokenizer.word_to_id["<end>"] == 2
    assert tokenizer.word_to_id["<unk>"] == 3


def test_vocab_respects_min_frequency():
    tokenizer = Tokenizer(min_word_freq=2)
    tokenizer.fit(["dog barks", "dog runs", "cat meows"])
    assert "dog" in tokenizer.word_to_id
    assert "barks" not in tokenizer.word_to_id
    assert "cat" not in tokenizer.word_to_id
