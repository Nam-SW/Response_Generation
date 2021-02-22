import tensorflow as tf

from modeling.utils import gelu, get_shape


class Attention(tf.keras.layers.Layer):
    def __init__(
        self,
        num_attention_heads: int,
        hidden_size: int,
    ):
        super().__init__()

        assert (
            hidden_size % num_attention_heads == 0
        ), f"hidden_size: {hidden_size}, num_attention_heads: {num_attention_heads}"
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.attention_head_size = int(hidden_size / num_attention_heads)

        param_size = self.num_attention_heads * self.attention_head_size
        self.query = tf.keras.layers.Dense(
            param_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name="query",
        )
        self.key = tf.keras.layers.Dense(
            param_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name="key",
        )
        self.value = tf.keras.layers.Dense(
            param_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name="value",
        )
        self.linear = tf.keras.layers.Dense(
            self.hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
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
        x = x / tf.math.sqrt(dk)

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
                self.hidden_size,
            ),
        )
        context = self.linear(context)

        return context, attention_weights

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_attention_heads": self.num_attention_heads,
                "hidden_size": self.hidden_size,
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
        self.selfattention = Attention(self.num_attention_heads, self.hidden_size)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)

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
        attention_output = self.selfattention(
            input_ids, attention_mask=attention_mask, training=training
        )[0]
        attention_output = self.dropout1(attention_output)
        attention_output = self.norm1(attention_output + input_ids)

        output = self.FFNN(attention_output)
        output = self.dropout2(output)
        output = self.norm2(output + attention_output)

        return output

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
        self.attention1 = Attention(self.num_attention_heads, self.hidden_size)
        self.attention2 = Attention(self.num_attention_heads, self.hidden_size)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(self.dropout_rate)

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
        )[0]
        attention_outputs_1 = self.dropout1(attention_outputs_1)
        attention_outputs_1 = self.norm1(attention_outputs_1 + input_ids)

        attention_outputs_2 = self.attention2(
            attention_outputs_1,
            k=encoder_output,
            v=encoder_output,
            attention_mask=padding_mask,
            training=training,
        )[0]
        attention_outputs_2 = self.dropout2(attention_outputs_2)
        attention_outputs_2 = self.norm2(attention_outputs_2 + attention_outputs_1)

        output = self.FFNN(attention_outputs_2)
        output = self.dropout3(output)
        output = self.norm3(output + attention_outputs_2)

        return output

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

        self.attention = Attention(self.attention_head, self.hidden_size)
        self.dense1 = tf.keras.layers.Dense(
            self.hidden_size,
            activation="relu",
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        )
        self.dense2 = tf.keras.layers.Dense(
            self.hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        )

    def call(self, q, k=None, v=None, attention_mask=None, training=False):
        k = q if k is None else k
        v = q if v is None else v
        if attention_mask is not None:
            attention_mask = (1.0 - attention_mask[:, tf.newaxis, tf.newaxis, :]) * -1e9

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
