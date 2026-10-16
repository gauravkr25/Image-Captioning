from __future__ import annotations

import tensorflow as tf


class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, warmup_steps, total_steps, name=None):
        super().__init__()
        self.peak_lr = float(peak_lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.name = name or "warmup_cosine"

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)

        warmup_scale = tf.maximum(warmup_steps, 1.0)
        warmup_progress = tf.minimum(step / warmup_scale, 1.0)
        decay_progress = (step - warmup_steps) / tf.maximum(total_steps - warmup_steps, 1.0)
        decay_progress = tf.clip_by_value(decay_progress, 0.0, 1.0)
        cosine = 0.5 * (1.0 + tf.cos(tf.constant(3.141592653589793, dtype=tf.float32) * decay_progress))
        return tf.where(step < warmup_steps, self.peak_lr * warmup_progress, self.peak_lr * cosine)

    def get_config(self):
        return {
            "peak_lr": self.peak_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "name": self.name,
        }


def masked_sparse_ce(y_true, y_pred, label_smoothing=0.0):
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.not_equal(y_true, 0)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    vocab_size = tf.shape(y_pred_masked)[-1]
    y_true_masked = tf.clip_by_value(y_true_masked, 0, vocab_size - 1)
    if label_smoothing > 0.0:
        smooth = label_smoothing / tf.cast(vocab_size, tf.float32)
        targets = tf.one_hot(y_true_masked, depth=vocab_size, dtype=tf.float32)
        smooth_targets = (1.0 - label_smoothing) * targets + smooth
        loss = -tf.reduce_sum(smooth_targets * tf.nn.log_softmax(y_pred_masked, axis=-1), axis=-1)
        return tf.reduce_mean(loss)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_masked, y_pred_masked, from_logits=True)
    return tf.reduce_mean(loss)


def masked_accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.not_equal(y_true, 0)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    y_pred_ids = tf.argmax(y_pred_masked, axis=-1)
    return tf.reduce_mean(tf.cast(tf.equal(y_true_masked, y_pred_ids), tf.float32))
