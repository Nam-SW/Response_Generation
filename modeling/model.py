from typing import Dict, Tuple

import tensorflow as tf

from modeling.attention import Attention, SelfAttention
from modeling.embedding import Embedding
from modeling.utils import get_shape


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
        FFNN_size: int,
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
        self.FFNN_size = FFNN_size

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

        self.FFNN = tf.keras.models.Sequential(
            layers=[
                tf.keras.layers.Dense(self.FFNN_size, activation="linear"),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.BatchNormalization(),
            ],
            name="FFNN",
        )
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
        input_contexts,
        shuffled_idx,
        # training=False
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
        E = self.concat(encoder_output)
        q = tf.concat(
            [encoder_output[shuffled_idx[i]] for i in range(batch_size)], axis=0
        )

        # decoder_attention = self.decoder_attention(q, k=E, v=E, training=training)[0]
        decoder_attention = self.decoder_attention(q, k=E, v=E)[0]
        ffnn = self.FFNN(decoder_attention)
        output = self.output_layer(ffnn)
        return output

    def call_utterance_order_recovery(
        self,
        input_contexts,
        # training=False
    ):
        # TODO: implement
        return None

    def call_masked_word_recovery(
        self,
        input_contexts,
        masked_utterance_idx,
        masked_word_idxes,
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
        E = self.concat(encoder_output)
        q = tf.concat(
            [encoder_output[masked_utterance_idx[i]] for i in range(batch_size)], axis=0
        )

        # decoder_attention = self.decoder_attention(q, k=E, v=E, training=training)[0]
        decoder_attention = self.decoder_attention(q, k=E, v=E)[0]
        ffnn = self.FFNN(decoder_attention)
        output = self.output_layer(ffnn)
        output_masked = [
            tf.gather(output[i], masked_word_idxes[i]) for i in range(batch_size)
        ]
        return output_masked

    def call_masked_utterance_recovery(
        self,
        input_contexts,
        masked_utterance_idx,
        # training=False
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
        E = self.concat(encoder_output)
        q = tf.concat(
            [encoder_output[masked_utterance_idx[i]] for i in range(batch_size)], axis=0
        )

        # decoder_attention = self.decoder_attention(q, k=E, v=E, training=training)[0]
        decoder_attention = self.decoder_attention(q, k=E, v=E)[0]
        ffnn = self.FFNN(decoder_attention)
        output = self.output_layer(ffnn)
        return output

    # def call_auxiliary_tasks(
    #     self,
    #     info_dict: Dict,
    #     input_contexts,
    #     # training=False
    # ):
    #     assert set(info_dict.keys()) == {
    #         "wor_shuffled_idx",
    #         "mwr_masked_utterance_idx",
    #         "mwr_masked_idxes",
    #         "mur_masked_utterance_idx",
    #     }, "argument is not required."

    #     return (
    #         self.call_word_order_recovery(
    #             shuffled_idx=info_dict["wor_shuffled_idx"],
    #             input_contexts=input_contexts,
    #             # training=training,
    #         ),
    #         # self.call_utterance_order_recovery(input_contexts, token_type_ids, training=training),
    #         self.call_masked_word_recovery(
    #             masked_utterance_idx=info_dict["mwr_masked_utterance_idx"],
    #             masked_idxes=info_dict["mwr_masked_idxes"],
    #             input_contexts=input_contexts,
    #             # training=training,
    #         ),
    #         self.call_masked_utterance_recovery(
    #             masked_utterance_idx=info_dict["mur_masked_utterance_idx"],
    #             input_contexts=input_contexts,
    #             # training=training,
    #         ),
    #     )

    def call(self, data: Tuple, training=False):
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

        if training:
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

    def convert_dict_to_tuple(self, dict_data: Dict):
        kw = dict_data.keys()

        # 변수 할당
        assert "input_contexts" in kw, "input_contexts  must require."
        input_contexts = dict_data["input_contexts"]
        assert "input_response" in kw, "input_response  must require."
        input_response = dict_data["input_response"]

        contexts_token_type_ids = dict_data.get("contexts_token_type_ids", None)
        response_token_type_ids = dict_data.get("response_token_type_ids", None)
        training = dict_data.get("training", False)

        return (
            input_contexts,
            input_response,
            contexts_token_type_ids,
            response_token_type_ids,
            training,
        )

    def predict(self, data: Dict):
        data = self.convert_dict_to_tuple(data)

        return super().predict(data)

    # def _repeats(tensor, repeats, axis):
    #     return repeat(tensor, repeats=repeats, axis=axis)

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
            "FFNN_size": self.FFNN_size,
        }
        return config
