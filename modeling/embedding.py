import numpy as np
import tensorflow as tf

from modeling.utils import get_shape, gather


class Embedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size, max_len, dropout_rate=0.2):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_len = max_len
        self.dropout_rate = dropout_rate

        self.position_embedding = self.positional_encoding()

        self.word_embedding = self.add_weight(
            "weight",
            shape=(self.vocab_size, self.embedding_size),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        )

        self.normalize = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def get_angles(self, pos, i):
        angles = 1 / tf.pow(
            10000, (2 * (i // 2)) / tf.cast(self.embedding_size, tf.float32)
        )
        return pos * angles

    def positional_encoding(self):
        angle_rads = self.get_angles(
            tf.range(self.vocab_size, dtype=tf.float32)[:, tf.newaxis],
            tf.range(self.embedding_size, dtype=tf.float32)[tf.newaxis, :],
        )

        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        angle_rads = np.zeros(angle_rads.shape)
        angle_rads[:, 0::2] = sines
        angle_rads[:, 1::2] = cosines
        pos_encoding = tf.constant(angle_rads)

        return tf.cast(pos_encoding[tf.newaxis, ...], dtype=tf.float32)

    def call(self, input_ids, token_type_ids=None):
        word_embedding = gather(self.word_embedding, input_ids)
        word_embedding *= tf.math.sqrt(tf.cast(self.embedding_size, tf.float32))
        if token_type_ids is None:
            segment_embedding = tf.fill(get_shape(word_embedding), 0.0)
        else:
            segment_embedding = tf.repeat(
                tf.cast(token_type_ids, dtype=word_embedding.dtype)[
                    :, :, tf.newaxis
                ],
                self.embedding_size,
                axis=-1,
            )

        input_len = get_shape(word_embedding)[1]
        # position_embedding = self.positional_encoding(input_len)
        embedding = (
            word_embedding
            + self.position_embedding[:, :input_len, :]
            + segment_embedding[:, :input_len, :]
        )
        embedding = self.normalize(embedding)
        embedding = self.dropout(embedding)

        return embedding

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embedding_size": self.embedding_size,
                "max_len": self.max_len,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config
