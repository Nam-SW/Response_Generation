from typing import Dict

import tensorflow as tf

from modeling.attention import (
    AttentionBlock,
    TransformerDecoder,
    TransformerEncoder,
)
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

        self.embedding = Embedding(
            self.vocab_size, self.embedding_size, self.max_len
        )
        self.dense = tf.keras.layers.Dense(self.hidden_size, use_bias=False)
        self.encoder_blocks = [
            AttentionBlock(
                self.hidden_size,
                self.attention_head,
                self.dropout_rate,
                name=f"attention_block_{i+1}",
            )
            for i in range(self.encoder_block)
        ]
        self.set_attention_mask = tf.keras.layers.Lambda(
            lambda x: tf.dtypes.cast((x >= 1), tf.float32)
        )
        self.decoder_block = AttentionBlock(
            self.hidden_size,
            self.attention_head,
            self.dropout_rate,
            name="decoder_block",
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

    def call_encoder(
        self, input_ids, token_type_ids, attention_mask, training=False
    ):
        embedding = self.embedding(input_ids, token_type_ids)
        embedding = self.dense(embedding, training=training)

        if attention_mask is None:
            attention_mask = tf.fill(get_shape(input_ids), 1.0)

        result = None
        for i, layer in enumerate(self.encoder_blocks):
            if result is None:
                result = layer(
                    embedding, attention_mask=attention_mask, training=training
                )

            else:
                result = layer(
                    result, attention_mask=attention_mask, training=training
                )

        return result

    def call_word_order_recovery(
        self,
        data: Dict,
        training=False,
    ):
        context_ids = data["context_ids"]
        context_token_type_ids = data.get("context_token_type_ids", None)
        context_attention_mask = data.get("context_attention_mask", None)

        encoder_output = self.call_encoder(
            context_ids,
            context_token_type_ids,
            self.set_attention_mask(context_ids)
            if context_attention_mask is None
            else context_attention_mask,
            training=training,
        )
        output = self.output_layer(encoder_output)
        return output

    def call_utterance_order_recovery(
        self,
        data: Dict,
        training=False,
    ):
        # TODO: implement
        return None

    def call_masked_content_recovery(
        self,
        data: Dict,
        training=False,
    ):
        context_ids = data["context_ids"]
        context_token_type_ids = data.get("context_token_type_ids", None)
        context_attention_mask = data.get("context_attention_mask", None)

        batch_size = get_shape(context_ids)[0]

        if context_token_type_ids is None:
            context_token_type_ids = self.get_token_type_ids(batch_size)
        context_encoder_output = [
            self.call_encoder(
                context_ids[:, i],
                context_token_type_ids[i],
                self.set_attention_mask(context_ids[:, i])
                if context_attention_mask is None
                else context_ids[:, i],
                training=training,
            )
            for i in range(self.utterance_size)
        ]
        E = tf.concat(context_encoder_output, axis=0)

        output = self.output_layer(E, training=True)
        shape = [batch_size, -1] + list(output.shape[1:])
        return tf.reshape(output, shape)

    def call_MLE(self, data, training=False):
        context_ids = data["context_ids"]
        context_token_type_ids = data.get("context_token_type_ids", None)
        context_attention_mask = data.get("context_attention_mask", None)

        response_ids = data.get("response_ids", None)
        response_token_type_ids = data.get("response_token_type_ids", None)
        response_attention_mask = data.get("response_attention_mask", None)
        if response_attention_mask is None:
            response_attention_mask = self.set_attention_mask(response_ids)

        batch_size = get_shape(context_ids)[0]

        # call
        if context_token_type_ids is None:
            context_token_type_ids = self.get_token_type_ids(batch_size)
        context_encoder_output = [
            self.call_encoder(
                context_ids[:, i],
                context_token_type_ids[i],
                self.set_attention_mask(context_ids[:, i])
                if context_attention_mask is None
                else context_attention_mask,
                training=training,
            )
            for i in range(self.utterance_size)
        ]
        response_encoder_output = self.call_encoder(
            response_ids,
            response_token_type_ids,
            response_attention_mask,
            training=training,
        )

        decoder_output = []
        for i in range(get_shape(response_encoder_output)[1]):
            E = tf.concat(
                context_encoder_output
                + [response_encoder_output[:, : i + 1, :]],
                axis=1,
            )

            temp = self.decoder_block(
                response_encoder_output[:, i, tf.newaxis],
                k=E,
                v=E,
                attention_mask=response_attention_mask[:, i, tf.newaxis],
                training=training,
            )
            decoder_output.append(temp)
        decoder_output = tf.concat(decoder_output, axis=1)

        output = self.output_layer(decoder_output)

        return output

    def generate(self, data):
        context_ids = data["context_ids"]
        context_token_type_ids = data.get("context_token_type_ids", None)
        context_attention_mask = data.get("context_attention_mask", None)

        response_ids = data.get("response_ids", None)
        response_token_type_ids = data.get("response_token_type_ids", None)
        response_attention_mask = data.get("response_attention_mask", None)
        if response_attention_mask is None:
            response_attention_mask = self.set_attention_mask(response_ids)

        batch_size = get_shape(context_ids)[0]

        # call
        if context_token_type_ids is None:
            context_token_type_ids = self.get_token_type_ids(batch_size)
        decoder_context_encoder_outputsoutput = [
            self.call_encoder(
                context_ids[:, i],
                context_token_type_ids[i],
                self.set_attention_mask(context_ids[:, i])
                if context_attention_mask is None
                else context_attention_mask,
            )
            for i in range(self.utterance_size)
        ]
        response_encoder_output = self.call_encoder(
            response_ids,
            response_token_type_ids,
            response_attention_mask,
        )

        E = tf.concat(
            decoder_context_encoder_outputsoutput
            + [response_encoder_output[:, :-1, :]],
            axis=1,
        )
        decoder_output = self.decoder_block(
            response_encoder_output[:, -1, tf.newaxis, :],
            k=E,
            v=E,
        )

        output = self.output_layer(decoder_output)

        return output

    def call(self, data: Dict, task="GENERATE", training=False):
        assert task in [
            "MLE",
            "WOR",
            "UOR",
            "MWR",
            "MUR",
            "GENERATE",
        ], "task is Literal['MLE', 'WOR', 'UOR', 'MWR', 'MUR', 'GENERATE']"

        data = data["MLE"]

        if task == "MLE":
            output = self.call_MLE(data, training=training)
        elif task == "WOR":
            output = self.call_word_order_recovery(data, training=training)
        elif task == "UOR":
            output = None
        elif task in ["MWR", "MUR"]:
            output = self.call_masked_content_recovery(data, training=training)
        elif task == "GENERATE":
            output = self.generate(data)

        return output

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


class Transformer(tf.keras.Model):
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
        super(Transformer, self).__init__(**kwargs)

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.utterance_size = utterance_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.attention_head = attention_head
        self.encoder_block = encoder_block
        self.dropout_rate = dropout_rate

        self.embedding = Embedding(
            self.vocab_size, self.hidden_size, self.max_len
        )
        self.encoders = [
            TransformerEncoder(
                self.hidden_size,
                self.attention_head,
                self.dropout_rate,
                name=f"encoder_{i}",
            )
            for i in range(encoder_block)
        ]
        self.decoder = TransformerDecoder(
            self.hidden_size, self.attention_head, self.dropout_rate
        )
        self.output_layer = tf.keras.layers.Dense(
            self.vocab_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=0.02
            ),
            activation="softmax",
        )

    def create_padding_mask(self, x):
        mask = tf.cast(tf.math.equal(x, 0), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, x):
        seq_len = tf.shape(x)[1]
        look_ahead_mask = 1 - tf.linalg.band_part(
            tf.ones((seq_len, seq_len)), -1, 0
        )
        padding_mask = self.create_padding_mask(x)
        return tf.maximum(look_ahead_mask, padding_mask)

    def call(self, data, task="MLE", training=False):
        assert task in [
            "MLE",
            "WOR",
            "UOR",
            "MWR",
            "MUR",
        ], "task is Literal['MLE', 'WOR', 'UOR', 'MWR', 'MUR']"

        context_ids = data["context_ids"]
        context_token_type_ids = data.get("context_token_type_ids", None)
        context_attention_mask = self.create_padding_mask(context_ids)

        context_embedding = self.embedding(context_ids, context_token_type_ids)
        encoder_output = context_embedding
        for layer in self.encoders:
            encoder_output = layer(
                encoder_output, context_attention_mask, training=training
            )

        if task == "MLE":
            response_ids = data.get("response_ids", None)
            response_token_type_ids = data.get("response_token_type_ids", None)
            response_attention_mask = self.create_look_ahead_mask(response_ids)

            response_embedding = self.embedding(
                response_ids, response_token_type_ids
            )
            decoder_output = self.decoder(
                response_embedding,
                encoder_output,
                padding_mask=context_attention_mask,
                look_ahead_mask=response_attention_mask,
                training=training,
            )

            output = self.output_layer(decoder_output)

        elif task in ["WOR", "MWR", "MUR"]:
            output = self.output_layer(encoder_output)

        elif task == "UOR":  # TODO: implement
            output = self.output_layer(encoder_output)

        return output
