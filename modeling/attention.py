import tensorflow as tf

from modeling.utils import gelu, get_shape


class SelfAttention(tf.keras.layers.Layer):
    def __init__(
        self, num_attention_heads: int, hidden_size: int, dropout_rate: float = 0.2
    ):
        super().__init__()

        assert (
            hidden_size % num_attention_heads == 0
        ), f"hidden_size: {hidden_size}, num_attention_heads: {num_attention_heads}"
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.dropout_rate = dropout_rate

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
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def transpose_for_scores(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_attention_heads, self.attention_head_size)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k=None, v=None, attention_mask=None, training=False):
        q = q
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
            # attention_mask = self.transpose_for_scores(attention_mask, batch_size)
            x *= attention_mask

        attention_weights = tf.nn.softmax(x, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)

        context = tf.matmul(attention_weights, value)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(
            context,
            (batch_size, -1, self.num_attention_heads * self.attention_head_size),
        )

        return context, attention_weights

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_attention_heads": self.num_attention_heads,
                "attention_head_size": self.attention_head_size,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class Attention(tf.keras.layers.Layer):
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
        self.selfattention = SelfAttention(num_attention_heads, hidden_size)
        self.dense = tf.keras.layers.Dense(
            self.hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        )
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.layernormal = tf.keras.layers.LayerNormalization()

        # Intermediate
        self.intermediate = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        )

        # output
        self.output_dense = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        )

    def call(self, input_ids, attention_mask=None, training=False):
        attention_outputs = self.selfattention(
            input_ids, attention_mask=attention_mask, training=training
        )
        x = self.dense(attention_outputs[0])
        x = self.dropout(x, training=training)
        x = self.layernormal(x + input_ids)

        intermediate_output = self.intermediate(x)
        intermediate_output = tf.keras.layers.Activation(gelu)(intermediate_output)

        x = self.output_dense(intermediate_output)
        x = self.dropout(x, training=training)
        x = self.layernormal(x + attention_outputs[0])

        return (x, attention_outputs[1])

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
