import tensorflow as tf


def get_shape(x):
    static = list(x.shape)
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def gelu(x):
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))
    return x * cdf


class FFNN(tf.keras.layers.Layer):
    def __init__(self, unit_size, dropout_rate=0.2):
        super().__init__()

        self.unit_size = unit_size
        self.dropout_rate = dropout_rate

        self.dense = tf.keras.layers.Dense(self.unit_size)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.normalize = tf.keras.layers.LayerNormalization()

    def call(self, data):
        x = self.dense(data)
        x = self.dropout(x)
        x = self.normalize(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "unit_size": self.unit_size,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config
