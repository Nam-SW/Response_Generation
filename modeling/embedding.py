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

        self.position_embedding = self.positional_encoding(self.max_len)

        self.word_embedding = self.add_weight(
            "weight",
            shape=(self.vocab_size, self.embedding_size),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        )

        self.normalize = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def get_angles(self, pos, i):
        angle_rates = 1 / np.power(
            10000, (2 * (i // 2)) / np.float32(self.embedding_size)
        )
        return pos * angle_rates

    def positional_encoding(self, max_len):
        angle_rads = self.get_angles(
            np.arange(max_len)[:, np.newaxis],
            np.arange(self.embedding_size)[np.newaxis, :],
        )

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, input_ids, token_type_ids=None):
        word_embedding = gather(self.word_embedding, input_ids)
        if token_type_ids is None:
            segment_embedding = tf.fill(get_shape(word_embedding), 0.0)
        else:
            segment_embedding = tf.repeat(
                tf.cast(token_type_ids, dtype=word_embedding.dtype)[:, :, tf.newaxis],
                self.embedding_size,
                axis=-1,
            )

        input_len = get_shape(word_embedding)[1]
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
