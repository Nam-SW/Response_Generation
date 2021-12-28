import json
import os
from typing import Dict

import tensorflow as tf
import transformers
from transformers import PretrainedConfig, TFAutoModel

from models.TransformerLayers import DecoderLayer, EncoderLayer
from models.UtilLayers import PositionalEmbedding


class Transformer(tf.keras.Model):
    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        vocab_size: int,
#         code_m: int,
        pe: int = 1000,
        rate: float = 0.1,
        pre_ln: bool = True,
    ):
        super(Transformer, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.vocab_size = vocab_size
#         self.code_m = code_m
        self.pe = pe
        self.rate = rate
        self.pre_ln = pre_ln

        self.embedding = PositionalEmbedding(vocab_size, embedding_size, pe)
        if embedding_size != hidden_size:
            self.embedding_intermediate = tf.keras.layers.Dense(hidden_size)
        self.encoders = [
            EncoderLayer(hidden_size, num_heads, rate, pre_ln)
            for _ in range(num_encoder_layers)
        ]
#         self.code_embedding = tf.keras.layers.Embedding(self.code_m, self.hidden_size)
        self.decoders = [
            DecoderLayer(hidden_size, num_heads, rate, pre_ln)
            for _ in range(num_decoder_layers)
        ]

        self.output_layer = tf.keras.layers.Dense(vocab_size, activation="linear")

    def create_look_ahead_mask(self, padding_mask):
        size = tf.shape(padding_mask)[-1]

        look_ahead_mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)

        combine_mask = tf.minimum(
            padding_mask[:, tf.newaxis, tf.newaxis, :],
            tf.cast(look_ahead_mask, padding_mask.dtype),
        )
        return combine_mask

    def dot_attention(self, q, k, v, mask=None):
        logits = tf.matmul(q, k, transpose_b=True)

        if mask is not None:
            logits += tf.cast((1 - mask[:, tf.newaxis, :]), tf.float32) * -1e9

        attention_weights = tf.nn.softmax(logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        return output

    def call(
        self,
        inputs: Dict,
        training=False,
    ):
        input_ids = inputs.get("input_ids", None)
        encoder_embed = inputs.get("input_embed", None)
        decoder_input_ids = inputs.get("decoder_input_ids", None)
        attention_mask = inputs.get("attention_mask", None)
        decoder_attention_mask = inputs.get("decoder_attention_mask", None)

        # check error
        assert (
            input_ids is not None or encoder_embed is not None
        ), "Either input_ids or encoder_embed must be required."
        assert (
            input_ids is None or encoder_embed is None
        ), "Only one of input_ids and encoder_embed must be entered."
        assert decoder_input_ids is not None, "deocder_input_ids must be required."

        # encoder
        if input_ids is not None:
            encoder_output = self.embedding(input_ids)
            if self.embedding_size != self.hidden_size:
                encoder_output = self.embedding_intermediate(encoder_output)
                
            for i in range(self.num_encoder_layers):
                encoder_output = self.encoders[i](encoder_output, attention_mask, training=training)
                
#             window = input_ids.shape[1]
#             encoder_embeds = []
#             for i in range(window):
#                 ids = input_ids[:, i, :]
#                 mask = attention_mask[:, i, :] if attention_mask is not None else None

#                 output = self.embedding(ids)
#                 if self.embedding_size != self.hidden_size:
#                     output = self.embedding_intermediate(output)

#                 for i in range(self.num_encoder_layers):
#                     output = self.encoders[i](output, mask, training=training)

#                 encoder_embeds.append(output)

#             encoder_embeds = tf.concat(encoder_embeds, axis=1)
#             attention_mask = (
#                 tf.reshape(attention_mask, encoder_embeds.shape[:2])
#                 if attention_mask is not None
#                 else None
#             )

#             codes = tf.range(self.code_m, dtype=tf.int32)
#             code_embeds = self.code_embedding(codes)

#             encoder_output = self.dot_attention(
#                 code_embeds,
#                 encoder_embeds,
#                 encoder_embeds,
#                 mask=attention_mask,
#             )

        elif encoder_embed is not None:
            encoder_output = encoder_embed

        # decoder
        decoder_output = self.embedding(decoder_input_ids)
        if self.embedding_size != self.hidden_size:
            decoder_output = self.embedding_intermediate(decoder_output)

        decoder_attention_mask = (
            None
            if decoder_attention_mask is None
            else self.create_look_ahead_mask(decoder_attention_mask)
        )

        for i in range(self.num_decoder_layers):
            decoder_output = self.decoders[i](
                decoder_output,
                encoder_output,
                decoder_attention_mask,
                attention_mask,
#                 None,  # encoder output을 하나로 합치면서 의미가 없어짐
                training=training,
            )

        output = self.output_layer(decoder_output)

        return (output, encoder_output)

    def get_config(self):
        return {
            "embedding_size": self.embedding_size,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "vocab_size": self.vocab_size,
#             "code_m": self.code_m,
            "pe": self.pe,
            "rate": self.rate,
            "pre_ln": self.pre_ln,
        }

    def _get_sample_data(self):
        sample_data = {
            "input_ids": tf.random.uniform(
                (1, 8), 0, self.vocab_size, dtype=tf.int64
            ),
            "decoder_input_ids": tf.random.uniform(
                (1, 1), 0, self.vocab_size, dtype=tf.int64
            ),
        }
        return sample_data

    def save(self, save_dir):
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(self.get_config(), f)

        self(self._get_sample_data())
        self.save_weights(os.path.join(save_dir, "model_weights.h5"))

        return os.listdir(save_dir)

    @classmethod
    def load(cls, save_dir):
        with open(os.path.join(save_dir, "config.json"), "r") as f:
            config = json.load(f)

        model = cls(**config)
        model(model._get_sample_data())
        model.load_weights(os.path.join(save_dir, "model_weights.h5"))

        return model


class PolyEncoder(tf.keras.Model):
    def __init__(
        self,
        model_name: str = None,
        model_config: str = None,
        model_class: str = None,
        from_pt=False,
        code_m: int = 64,
        rate: float = 0.2,
    ):
        super(PolyEncoder, self).__init__()

        self.model_name = model_name
        self.model_config = model_config
        self.model_class = model_class
        self.from_pt = from_pt
        self.code_m = code_m
        self.rate = rate

        if model_name is not None:
            self.model = TFAutoModel.from_pretrained(model_name, from_pt=self.from_pt)
            self.model_name = None
            self.model_config = self.model.config
            self.model_class = self.model.__class__.__name__
        elif model_config is not None and model_class is not None:
            if isinstance(model_config, dict):
                model_config = PretrainedConfig.from_dict(model_config)
            elif not isinstance(model_config, PretrainedConfig):
                raise TypeError("model_config must be either dict or PretrainedConfig.")
            self.model = getattr(transformers, model_class)(model_config)
        else:
            raise KeyError("model_name or model_config, model_class must be required.")

        self.code_embedding = tf.keras.layers.Embedding(
            self.code_m, self.model.config.hidden_size
        )

    def loss(self, y, pred):
        mask = tf.eye(tf.shape(pred)[0], dtype=pred.dtype)
        loss = -tf.reduce_sum(tf.nn.log_softmax(pred, axis=-1) * mask)
        loss = tf.reduce_mean(loss)

        return loss

    def dot_attention(self, q, k, v, mask=None):
        logits = tf.matmul(q, k, transpose_b=True)

        if mask is not None:
            logits += tf.cast((1 - mask[:, tf.newaxis, :]), tf.float32) * -1e9

        attention_weights = tf.nn.softmax(logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        return output

    def get_embeds(self, input_ids, attention_mask=None, training=False):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, training=training
        )[0]

    def call(
        self,
        context_input_ids=None,
        context_input_embeds=None,
        candidate_input_ids=None,
        context_mask=None,
        candidate_mask=None,
        labels=None,
        training=False,
        **kwargs,
    ):
        # context - candidate == N - 1?
        assert (
            context_input_ids is not None or context_input_embeds is not None
        ), "Either context_input_ids or context_input_embeds must be required."
        assert candidate_input_ids is not None, "candidate_input_ids must be required."

        # context_embeds
        context_embeds = context_input_embeds
        if context_embeds is None:
            context_embeds = self.get_embeds(context_input_ids, training=training)

        codes = tf.range(self.code_m, dtype=tf.int32)
        code_embeds = self.code_embedding(codes)
        context_embeds = self.dot_attention(
            code_embeds, context_embeds, context_embeds, mask=context_mask
        )
        context_mask = tf.ones(context_embeds.shape[:-1])
        # context_mask = tf.ones(tf.shape(context_embeds)[:-1])

        # candidate_embeds
        if labels is not None:  # on training
            candidate_input_ids = candidate_input_ids[:, :1, :]
            if candidate_mask is not None:
                candidate_mask = candidate_mask[:, :1, :]

        batch_size, cand_n, seq_len = candidate_input_ids.shape
        # batch_size, cand_n, seq_len = tf.shape(candidate_input_ids)
        candidate_input_ids = tf.reshape(
            candidate_input_ids, (batch_size * cand_n, seq_len)
        )

        candidate_embeds = self.get_embeds(candidate_input_ids, training=training)[
            :, 0, :
        ]
        candidate_embeds = tf.reshape(candidate_embeds, (batch_size, cand_n, -1))
        if labels is not None:
            candidate_embeds = tf.repeat(
                tf.transpose(candidate_embeds, [1, 0, 2]), batch_size, axis=0
            )

        # combine
        combine_embeds = self.dot_attention(
            candidate_embeds, context_embeds, context_embeds, mask=context_mask
        )

        combine_output = tf.reduce_sum(combine_embeds * candidate_embeds, axis=-1)

        if labels is not None:
            return (combine_output, self.loss(labels, combine_output))
        else:
            return (combine_output,)

    def get_config(self):
        model_config = self.model_config.to_dict()
        return {
            "model_name": self.model_name,
            "model_config": model_config,
            "model_class": self.model_class,
            "from_pt": self.from_pt,
            "code_m": self.code_m,
            "rate": self.rate,
        }

    # def _get_sample_data(self):
    #     sample_data = {
    #         "context_input_ids": tf.random.uniform(
    #             (1, 8), 0, self.model_config.vocab_size, dtype=tf.int64
    #         ),
    #         "candidate_input_ids": tf.random.uniform(
    #             (1, 8), 0, self.model_config.vocab_size, dtype=tf.int64
    #         ),
    #     }
    #     return sample_data

    # def save(self, save_dir):
    #     if not os.path.isdir(save_dir):
    #         os.mkdir(save_dir)

    #     with open(os.path.join(save_dir, "config.json"), "w") as f:
    #         json.dump(self.get_config(), f)

    #     self(**self._get_sample_data())
    #     self.save_weights(os.path.join(save_dir, "model_weights.h5"))

    #     return os.listdir(save_dir)

    # @classmethod
    # def load(cls, save_dir):
    #     with open(os.path.join(save_dir, "config.json"), "r") as f:
    #         config = json.load(f)

    #     model = cls(**config)
    #     model(**model._get_sample_data())
    #     model.load_weights(os.path.join(save_dir, "model_weights.h5"))

    #     return model
