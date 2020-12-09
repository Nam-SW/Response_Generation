from typing import Tuple

import tensorflow as tf

from modeling.attention import Attention, SelfAttention
from modeling.embedding import Embedding
from modeling.utils import get_shape, FFNN


class DialogWithAuxility(tf.keras.Model):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        utterance_size: int,
        embedding_size: int,
        hidden_size: int,
        attention_head: int,
        encoder_block: int,
        dropout_rate: float,
        **kwargs,
    ):
        super(DialogWithAuxility, self).__init__(**kwargs)

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.utterance_size = utterance_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.attention_head = attention_head
        self.encoder_block = encoder_block
        self.dropout_rate = dropout_rate

        self.embedding = Embedding(self.vocab_size, self.embedding_size, self.max_len)
        self.dense = tf.keras.layers.Dense(self.hidden_size, use_bias=False)
        self.attention_blocks = [
            Attention(
                self.hidden_size,
                self.attention_head,
                self.dropout_rate,
                name=f"attention_block_{i+1}",
            )
            for i in range(self.encoder_block)
        ]
        self.set_attention_mask = tf.keras.layers.Lambda(
            lambda x: tf.dtypes.cast((x > 1), tf.float32)
        )
        self.decoder_attention = SelfAttention(self.attention_head, self.hidden_size)

        self.concat = tf.keras.layers.Concatenate(axis=1)

        # self.FFNN_dense = tf.keras.layers.Dense(self.hidden_size, activation="linear")
        # self.FFNN_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        # self.FFNN_normalize = tf.keras.layers.LayerNormalization()
        self.FFNN = FFNN(self.hidden_size, self.dropout_rate)
        self.output_layer = tf.keras.layers.Dense(
            self.vocab_size, use_bias=False, activation="softmax"
        )

    def get_token_type_ids(self, batch_size):
        ids = [1.0, 0.0] if self.utterance_size % 2 else [0.0, 1.0]
        return [
            tf.fill((batch_size, self.max_len), ids[i % 2])
            for i in range(self.utterance_size)
        ]

    # def call_blocks(self, Input, blocks, mask=None, training=False):
    def call_blocks(self, Input, blocks, mask=None):
        if mask is None:
            mask = tf.fill(get_shape(Input)[:-1], 1.0)
        mask = (1.0 - mask[:, tf.newaxis, tf.newaxis, :]) * -10000.0

        result = None
        for i, layer in enumerate(blocks):
            if result is None:
                result, hidden_state = layer(Input, mask)
                # result, hidden_state = layer(Input, mask, training=training)

            else:
                # result, hidden_state = layer(result, training=training)
                result, hidden_state = layer(result)

        return (result, hidden_state)

    def call_encoder(self, input_ids, token_type_ids, attention_blocks):
        # def call_encoder(self, input_ids, token_type_ids, attention_blocks, training=False):
        embedding = self.embedding(input_ids, token_type_ids)
        embedding = self.dense(embedding)
        masks = self.set_attention_mask(input_ids)
        output = self.call_blocks(embedding, attention_blocks, masks)
        # output = self.call_blocks(embedding, attention_blocks, masks, training=training)

        return output

    def call_word_order_recovery(
        self,
        input_context,
        # training=False
    ):
        encoder_output = self.call_encoder(
            input_context,
            None,
            self.attention_blocks,
            # training=training,
        )[0]
        output = self.output_layer(encoder_output)
        return output

    def call_utterance_order_recovery(
        self,
        input_contexts,
        # training=False
    ):
        # TODO: implement
        return None

    def call_masked_content_recovery(
        self,
        input_contexts,
        # training=False,
    ):
        batch_size = get_shape(input_contexts)[0]

        contexts_token_type_ids = self.get_token_type_ids(batch_size)
        encoder_output = [
            self.call_encoder(
                input_contexts[:, i],
                contexts_token_type_ids[i],
                self.attention_blocks,
                # training=training,
            )[0]
            for i in range(self.utterance_size)
        ]
        E = tf.concat(encoder_output, axis=0)

        output = self.output_layer(E)
        shape = [batch_size, -1] + list(output.shape[1:])
        return tf.reshape(output, shape)

    def call_MLE(self, data, return_all_sequences=False):
        (
            input_contexts,
            input_response,
            # training,
        ) = data

        assert (
            len(get_shape(input_contexts)) == 3
        ), f"input ids ndim must be 3, but inputed {input_contexts.ndim} dim tensor."
        assert (
            get_shape(input_contexts)[1] == self.utterance_size
        ), f"A total of {self.utterance_size} utterances must be entered."

        batch_size = get_shape(input_contexts)[0]

        # call
        contexts_token_type_ids = self.get_token_type_ids(batch_size)
        contexts_encoder_outputs = [
            self.call_encoder(
                input_contexts[:, i],
                contexts_token_type_ids[i],
                self.attention_blocks,
                # training=training,
            )[0]
            for i in range(self.utterance_size)
        ]
        response_encoder_output = self.call_encoder(
            input_response,
            tf.fill((batch_size, self.max_len), 0),
            self.attention_blocks,
            # training=training,
        )[0]

        if return_all_sequences:
            decoder_attention = []
            for i in range(1, get_shape(response_encoder_output)[1]):
                E = self.concat(
                    contexts_encoder_outputs + [response_encoder_output[:, :i, :]]
                )

                temp = self.decoder_attention(
                    response_encoder_output[:, i : i + 1, :],
                    k=E,
                    v=E,
                    # training=training,
                )[0]
                decoder_attention.append(temp)
            decoder_attention = self.concat(decoder_attention)

        else:
            E = self.concat(
                contexts_encoder_outputs + [response_encoder_output[:, :-1, :]]
            )

            decoder_attention = self.decoder_attention(
                response_encoder_output[:, -1:, :],
                k=E,
                v=E,
                # training=training
            )[0]

        decoder_FFNN = self.FFNN(decoder_attention)
        output = self.output_layer(decoder_FFNN)

        return output

    def call(
        self,
        data: Tuple,
        return_all_sequences=False,
        task="MLE",
        # task: Literal["MLE", "WOR", "UOR", "MCR"] = "MLE",
    ):
        assert task in [
            "MLE",
            "WOR",
            "UOR",
            "MCR",
        ], "task is Literal['MLE', 'WOR', 'UOR', 'MCR']"

        if task == "MLE":
            pred = self.call_MLE(data, return_all_sequences)
        elif task == "WOR":
            pred = self.call_word_order_recovery(data)
        elif task == "UOR":
            pred = self.call_utterance_order_recovery(data)
        elif task == "MCR":
            pred = self.call_masked_content_recovery(data)
        return pred

    def get_config(self):
        config = {
            "vocab_size": self.vocab_size,
            "max_len": self.max_len,
            "utterance_size": self.utterance_size,
            "embedding_size": self.embedding_size,
            "hidden_size": self.hidden_size,
            "attention_head": self.attention_head,
            "encoder_block": self.encoder_block,
            "dropout_rate": self.dropout_rate,
        }
        return config
