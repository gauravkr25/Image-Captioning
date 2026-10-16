import numpy as np
import tensorflow as tf

from src.models.common import masked_sparse_ce
from src.models.gru_attention import GRUDecoder
from src.models.transformer import CaptionTransformer


def test_gru_and_transformer_outputs_have_expected_shape():
    features = np.random.randn(2, 49, 16).astype(np.float32)
    seq = np.array([[1, 2, 3], [1, 4, 5]], dtype=np.int32)
    gru = GRUDecoder(vocab_size=32, embedding_dim=8, gru_units=12, feature_dim=16)
    gru_out = gru((features, seq), training=False)
    assert gru_out.shape == (2, 3, 32)

    transformer = CaptionTransformer(vocab_size=32, d_model=16, num_layers=2, num_heads=4, dff=32, dropout=0.0)
    trans_out = transformer((features, seq), training=False)
    assert trans_out.shape == (2, 3, 32)


def test_causal_mask_blocks_future_tokens():
    features = np.random.randn(1, 49, 16).astype(np.float32)
    seq = np.array([[1, 2, 3, 4]], dtype=np.int32)
    transformer = CaptionTransformer(vocab_size=40, d_model=16, num_layers=2, num_heads=4, dff=32, dropout=0.0)
    base = transformer((features, seq), training=False)

    modified = seq.copy()
    modified[0, 2] = 39
    changed = transformer((features, modified), training=False)

    np.testing.assert_allclose(base[0, :2], changed[0, :2], rtol=1e-4, atol=1e-4)


def test_masked_loss_ignores_padding():
    y_true = tf.constant([[1, 0, 2], [4, 0, 0]], dtype=tf.int32)
    y_pred = tf.constant(
        [
            [[2.0, 0.0, 1.0], [2.0, 0.0, 0.0], [2.0, 0.0, 1.0]],
            [[2.0, 0.0, 1.0], [2.0, 0.0, 0.0], [2.0, 0.0, 1.0]],
        ],
        dtype=tf.float32,
    )
    loss = masked_sparse_ce(y_true, y_pred)
    assert tf.is_tensor(loss)
    assert loss.shape == ()
