from __future__ import annotations

import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def call(self, x):
        seq_len = tf.shape(x)[-2]
        positions = tf.cast(tf.range(seq_len), tf.float32)
        inv_timescale = 1.0 / (10000 ** (tf.range(0, self.d_model, 2, dtype=tf.float32) / tf.cast(self.d_model, tf.float32)))
        angle_rates = positions[:, None] * inv_timescale[None, :]
        angles = tf.concat([angle_rates, angle_rates], axis=-1)
        pos_encoding = tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)
        if self.d_model % 2 == 1:
            pos_encoding = tf.pad(pos_encoding, [[0, 0], [0, 1]])
        return x + pos_encoding[None, :, : self.d_model]


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.self_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout)
        self.cross_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn1 = tf.keras.layers.Dense(dff, activation="relu")
        self.ffn2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, features, training=None):
        causal_mask = self._causal_mask(tf.shape(x)[1])
        attn1 = self.self_attention(query=x, value=x, key=x, attention_mask=causal_mask, training=training, return_attention_scores=False)
        x = self.norm1(x + attn1)
        attn2 = self.cross_attention(query=x, value=features, key=features, training=training, return_attention_scores=False)
        x = self.norm2(x + attn2)
        ffn = self.ffn2(self.ffn1(x))
        x = self.norm3(x + self.dropout(ffn, training=training))
        return x

    def _causal_mask(self, size):
        lower = tf.linalg.band_part(tf.ones((size, size), dtype=tf.bool), -1, 0)
        return lower[None, None, :, :]


class CaptionTransformer(tf.keras.Model):
    def __init__(self, vocab_size, d_model=256, num_layers=3, num_heads=4, dff=512, dropout=0.1, feature_dim=1280):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.dropout = dropout
        self.feature_dim = feature_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.image_proj = tf.keras.layers.Dense(d_model)
        self.image_pos = PositionalEncoding(d_model)
        self.blocks = [DecoderBlock(d_model, num_heads, dff, dropout) for _ in range(num_layers)]
        self.output_layer = tf.keras.layers.Dense(vocab_size)
        self.dropout_layer = tf.keras.layers.Dropout(dropout)

    def build(self, input_shape):
        features_shape, decoder_shape = input_shape
        self.embedding.build((None, decoder_shape[1]))
        self.image_proj.build((None, features_shape[1], features_shape[2]))
        self.output_layer.build((None, decoder_shape[1], self.d_model))
        self.built = True

    def call(self, inputs, training=None):
        features, input_ids = inputs
        features = self.image_proj(features)
        features = self.image_pos(features)
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)
        x = self.dropout_layer(x, training=training)
        for block in self.blocks:
            x = block(x, features, training=training)
        logits = self.output_layer(x)
        return logits

    def get_config(self):
        config = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout": self.dropout,
            "feature_dim": self.feature_dim,
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
