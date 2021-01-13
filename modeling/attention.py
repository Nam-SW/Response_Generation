import tensorflow as tf

from modeling.utils import gelu, get_shape


class Attention(tf.keras.layers.Layer):
    def __init__(
        self,
        num_attention_heads: int,
        hidden_size: int,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        assert (
            hidden_size % num_attention_heads == 0
        ), f"hidden_size: {hidden_size}, num_attention_heads: {num_attention_heads}"
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.dropout_rate = dropout_rate

        param_size = self.num_attention_heads * self.attention_head_size
        self.query = tf.keras.layers.Dense(
            param_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=0.02
            ),
            name="query",
        )
        self.key = tf.keras.layers.Dense(
            param_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=0.02
            ),
            name="key",
        )
        self.value = tf.keras.layers.Dense(
            param_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=0.02
            ),
            name="value",
        )
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.linear = tf.keras.layers.Dense(
            self.hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=0.02
            ),
        )

    def transpose_for_scores(self, x, batch_size):
        x = tf.reshape(
            x,
            (
                batch_size,
                -1,
                self.num_attention_heads,
                self.attention_head_size,
            ),
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k=None, v=None, attention_mask=None, training=False):
        k = q if k is None else k
        v = q if v is None else v

        batch_size = get_shape(q)[0]
        query = self.transpose_for_scores(self.query(q), batch_size)
        key = self.transpose_for_scores(self.key(k), batch_size)
        value = self.transpose_for_scores(self.value(v), batch_size)

        # scaled dot attention
        x = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(get_shape(key)[-1], x.dtype)
        x /= tf.math.sqrt(dk)

        if attention_mask is not None:
            x += attention_mask * -1e9

        attention_weights = tf.nn.softmax(x, axis=-1)

        context = tf.matmul(attention_weights, value)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(
            context,
            (
                batch_size,
                -1,
                self.num_attention_heads * self.attention_head_size,
            ),
        )
        context = self.linear(context)
        context = self.dropout(context, training=training)

        return context, attention_weights

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_attention_heads": self.num_attention_heads,
                "hidden_size": self.hidden_size,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout_rate: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate

        # Attention
        self.selfattention = Attention(
            self.num_attention_heads, self.hidden_size, self.dropout_rate
        )
        self.layernormal = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self.FFNN = tf.keras.models.Sequential(
            layers=[
                tf.keras.layers.Dense(
                    self.hidden_size * 2,
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(
                        stddev=0.02
                    ),
                    activation="relu",
                ),
                tf.keras.layers.Dense(
                    self.hidden_size,
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(
                        stddev=0.02
                    ),
                ),
            ]
        )

    def call(self, input_ids, attention_mask, training=False):
        attention_outputs = self.selfattention(
            input_ids, attention_mask=attention_mask, training=training
        )
        x = self.layernormal(attention_outputs[0] + input_ids)

        ffnn = self.FFNN(x)
        ffnn = self.dropout(ffnn)
        x = self.layernormal(x + ffnn)

        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_attention_heads": self.num_attention_heads,
                "hidden_size": self.hidden_size,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout_rate: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate

        # Attention
        self.attention1 = Attention(
            self.num_attention_heads, self.hidden_size, self.dropout_rate
        )
        self.attention2 = Attention(
            self.num_attention_heads, self.hidden_size, self.dropout_rate
        )
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.layernormal = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.FFNN = tf.keras.models.Sequential(
            layers=[
                tf.keras.layers.Dense(
                    self.hidden_size * 2,
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(
                        stddev=0.02
                    ),
                    activation="relu",
                ),
                tf.keras.layers.Dense(
                    self.hidden_size,
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(
                        stddev=0.02
                    ),
                ),
            ]
        )

    def call(
        self,
        input_ids,
        encoder_output,
        padding_mask,
        look_ahead_mask,
        training=False,
    ):
        attention_outputs_1 = self.attention1(
            input_ids, attention_mask=look_ahead_mask, training=training
        )
        x = self.layernormal(attention_outputs_1[0] + input_ids)

        attention_outputs_2 = self.attention2(
            x,
            k=encoder_output,
            v=encoder_output,
            attention_mask=padding_mask,
            training=training,
        )
        x = self.layernormal(attention_outputs_2[0] + input_ids)

        ffnn = self.FFNN(x)
        x = self.dropout(ffnn)
        x = self.layernormal(x + ffnn)

        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_attention_heads": self.num_attention_heads,
                "hidden_size": self.hidden_size,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_size, attention_head, dropout_rate, **kwargs):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.attention_head = attention_head
        self.dropout_rate = dropout_rate

        self.attention = Attention(
            self.attention_head, self.hidden_size, self.dropout_rate
        )
        self.dense1 = tf.keras.layers.Dense(
            self.hidden_size,
            activation="relu",
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=0.02
            ),
        )
        self.dense2 = tf.keras.layers.Dense(
            self.hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=0.02
            ),
        )

    def call(self, q, k=None, v=None, attention_mask=None, training=False):
        k = q if k is None else k
        v = q if v is None else v
        if attention_mask is not None:
            attention_mask = (
                1.0 - attention_mask[:, tf.newaxis, tf.newaxis, :]
            ) * -1e9

        x = self.attention(
            q, k=k, v=v, attention_mask=attention_mask, training=training
        )[0]
        x = self.dense1(x, training=training)
        x = self.dense2(x, training=training)

        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "attention_head": self.attention_head,
                "hidden_size": self.hidden_size,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config
