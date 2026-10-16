from __future__ import annotations

import tensorflow as tf


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

        self.W_h = tf.keras.layers.Dense(units, use_bias=False)
        self.W_f = tf.keras.layers.Dense(units, use_bias=False)
        self.v = tf.keras.layers.Dense(1)

    def call(self, features, hidden_state):
        hidden_state = tf.expand_dims(hidden_state, axis=1)

        hidden_with_time = self.W_h(hidden_state)
        features_projection = self.W_f(features)

        score = self.v(
            tf.nn.tanh(features_projection + hidden_with_time)
        )

        scores = tf.squeeze(score, axis=-1)
        attention = tf.nn.softmax(scores, axis=-1)

        context = tf.reduce_sum(
            attention[:, :, None] * features,
            axis=1,
        )

        return context, attention


class GRUDecoder(tf.keras.Model):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        gru_units=256,
        feature_dim=1280,
        dropout=0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.gru_units = gru_units
        self.feature_dim = feature_dim

        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim
        )

        self.attention = BahdanauAttention(gru_units)
        self.init_state = tf.keras.layers.Dense(gru_units)

        self.gru = tf.keras.layers.GRU(
            gru_units,
            return_sequences=True,
            return_state=True,
        )

        self.dropout = tf.keras.layers.Dropout(dropout)
        self.output_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, training=None):
        features, decoder_input = inputs

        emb = self.embedding(decoder_input)
        sequence_length = tf.shape(decoder_input)[1]

        mean_features = tf.reduce_mean(features, axis=1)
        initial_hidden = self.init_state(mean_features)

        outputs = tf.TensorArray(
            dtype=tf.float32,
            size=sequence_length,
            clear_after_read=False,
        )

        def condition(timestep, hidden_state, output_array):
            return timestep < sequence_length

        def body(timestep, hidden_state, output_array):
            token_emb = emb[:, timestep, :]

            context, _ = self.attention(features, hidden_state)

            concat = tf.concat(
                [context, token_emb],
                axis=-1,
            )

            gru_out, new_hidden = self.gru(
                tf.expand_dims(concat, axis=1),
                initial_state=hidden_state,
                training=training,
            )

            gru_out = tf.squeeze(gru_out, axis=1)
            gru_out = self.dropout(
                gru_out,
                training=training,
            )

            logits = self.output_layer(gru_out)
            output_array = output_array.write(timestep, logits)

            return timestep + 1, new_hidden, output_array

        _, _, outputs = tf.while_loop(
            condition,
            body,
            (
                tf.constant(0),
                initial_hidden,
                outputs,
            ),
            parallel_iterations=1,
        )

        logits = outputs.stack()
        logits = tf.transpose(logits, perm=[1, 0, 2])

        return logits

    def decode_step(self, token, state, features):
        token = tf.convert_to_tensor(token, dtype=tf.int32)
        features = tf.convert_to_tensor(
            features, dtype=tf.float32
        )

        if features.shape.rank == 2:
            features = features[None, ...]

        if state is None:
            mean_features = tf.reduce_mean(features, axis=1)
            hidden_state = self.init_state(mean_features)
        else:
            hidden_state = state

        token_emb = self.embedding(tf.reshape(token, [-1]))

        context, attention = self.attention(
            features, hidden_state
        )

        concat = tf.concat(
            [context, token_emb],
            axis=-1,
        )

        gru_out, new_hidden = self.gru(
            tf.expand_dims(concat, axis=1),
            initial_state=hidden_state,
        )

        logits = self.output_layer(
            tf.squeeze(gru_out, axis=1)
        )

        return logits, new_hidden, attention

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "gru_units": self.gru_units,
            "feature_dim": self.feature_dim,
            "dropout": self.dropout.rate,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
