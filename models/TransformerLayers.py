import tensorflow as tf

from models.UtilLayers import FFNN, MultiHeadAttention


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, rate=0.1, pre_ln=True):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.rate = rate
        self.pre_ln = pre_ln

        # multi-head attention
        self.mha_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.mha_dropout = tf.keras.layers.Dropout(rate)

        # ffnn
        self.ffnn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffnn = FFNN(d_model, d_model * 4, "relu", rate)
        self.ffnn_dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, mask=None, training=False):
        # extended_mask
        extended_mask = mask[:, tf.newaxis, tf.newaxis, :] if mask is not None else None

        # multi-head attention
        if self.pre_ln:
            mha_output, _ = self.mha(x, x, x, extended_mask)
            mha_output = self.mha_norm(mha_output + x, training=training)
            mha_output = self.mha_dropout(mha_output, training=training)
        else:
            mha_output = self.mha_norm(x, training=training)
            mha_output, _ = self.mha(mha_output, mha_output, mha_output, extended_mask)
            mha_output = self.mha_dropout(mha_output + x, training=training)

        # ffnn
        if self.pre_ln:
            ffnn_output = self.ffnn(mha_output, training=training)
            ffnn_output = self.ffnn_norm(ffnn_output + mha_output, training=training)
            ffnn_output = self.ffnn_dropout(ffnn_output, training=training)
        else:
            ffnn_output = self.ffnn_norm(mha_output, training=training)
            ffnn_output = self.ffnn(ffnn_output, training=training)
            ffnn_output = self.ffnn_dropout(ffnn_output + mha_output, training=training)

        return ffnn_output

    def get_config(self):
        return {
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "rate": self.rate,
            "pre_ln": self.pre_ln,
        }


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, rate=0.1, pre_ln=True):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.rate = rate
        self.pre_ln = pre_ln

        # cross multi-head attention with encoder
        self.mha1_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha1_dropout = tf.keras.layers.Dropout(rate)

        # cross multi-head attention with encoder
        self.mha2_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.mha2_dropout = tf.keras.layers.Dropout(rate)

        # ffnn
        self.ffnn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffnn = FFNN(d_model, d_model * 2, "relu", rate)
        self.ffnn_dropout = tf.keras.layers.Dropout(rate)

    def call(
        self,
        x,
        enc_output,
        look_ahead_mask,
        padding_mask,
        training=False,
    ):
        # extended_mask
        extended_look_ahead_mask = look_ahead_mask

        extended_padding_mask = (
            padding_mask[:, tf.newaxis, tf.newaxis, :]
            if padding_mask is not None
            else None
        )

        # cross multi-head attention with encoder
        if self.pre_ln:
            mha1_output, _ = self.mha1(x, x, x, extended_look_ahead_mask)
            mha1_output = self.mha1_norm(mha1_output + x, training=training)
            mha1_output = self.mha1_dropout(mha1_output, training=training)
        else:
            mha1_output = self.mha1_norm(x, training=training)
            mha1_output, _ = self.mha1(
                mha1_output, mha1_output, mha1_output, extended_look_ahead_mask
            )
            mha1_output = self.mha1_dropout(mha1_output + x, training=training)

        # cross multi-head attention with encoder
        if self.pre_ln:
            mha2_output, _ = self.mha2(
                mha1_output, enc_output, enc_output, extended_padding_mask
            )
            mha2_output = self.mha2_norm(mha2_output + mha1_output, training=training)
            mha2_output = self.mha2_dropout(mha2_output, training=training)
        else:
            mha2_output = self.mha2_norm(mha1_output, training=training)
            mha2_output, _ = self.mha2(
                mha2_output, enc_output, enc_output, extended_padding_mask
            )
            mha2_output = self.mha2_dropout(
                mha2_output + mha1_output, training=training
            )

        # ffnn
        if self.pre_ln:
            ffnn_output = self.ffnn(mha2_output, training=training)
            ffnn_output = self.ffnn_norm(ffnn_output + mha2_output, training=training)
            ffnn_output = self.ffnn_dropout(ffnn_output, training=training)
        else:
            ffnn_output = self.ffnn_norm(mha2_output, training=training)
            ffnn_output = self.ffnn(ffnn_output, training=training)
            ffnn_output = self.ffnn_dropout(
                ffnn_output + mha2_output, training=training
            )

        return ffnn_output

    def get_config(self):
        return {
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "rate": self.rate,
            "pre_ln": self.pre_ln,
        }
