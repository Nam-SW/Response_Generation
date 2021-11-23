import numpy as np
import tensorflow as tf


class Generator:
    def __init__(
        self,
        model,
        tokenizer,
        model_token,
        model_max_len=128,
        window=3,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_token = model_token
        self.model_max_len = model_max_len
        self.window = window

    def start_setting(
        self,
        input_text=None,
        input_ids=None,
        decoder_ids=None,
    ):
        assert (
            input_ids is not None or input_text is not None
        ), "input_text or input_ids must be required."
        assert (
            input_ids is None or input_text is None
        ), "Only one of input_text and input_ids is required."

        if input_text is not None:
            input_text = ["" for _ in range(self.window - len(input_text))] + input_text

            input_ids = self.tokenizer(
                input_text,
                max_length=self.model_max_len,
                padding="max_length",
                truncation=True,
                return_tensors="tf",
            )["input_ids"][None, :, :]
        elif input_ids is not None:
            input_ids = tf.constant(input_ids, dtype=tf.int32)
            if input_ids.ndim != 3:
                raise ValueError(
                    "ndim of input_ids must be 3, but given " + str(input_ids.ndim)
                )

        if decoder_ids is None:
            decoder_ids = [self.tokenizer.bos_token_id, self.model_token]

        if isinstance(decoder_ids, np.ndarray):
            decoder_ids = decoder_ids.tolist()

        return input_ids, decoder_ids

    def generate_random_sampling(
        self,
        input_text=None,
        input_ids=None,
        decoder_ids=None,
        max_length=128,
        temperature=1.0,
        top_k=1,
    ):
        input_ids, decoder_ids = self.start_setting(
            input_text=input_text,
            input_ids=input_ids,
            decoder_ids=decoder_ids,
        )

        input_embed = None
        for _ in range(max_length - 1):
            inputs = {
                "input_ids": input_ids if input_embed is None else None,
                "input_embed": input_embed,
                "decoder_input_ids": tf.constant([decoder_ids], tf.int32),
            }
            last_pred, input_embed = self.model(inputs)
            last_pred = last_pred[0, -1, :].numpy() / float(temperature)

            if top_k == 1:
                candidate = int(np.argmax(last_pred, axis=-1))
            elif top_k >= 2:
                idx = last_pred.argsort()[: -(top_k + 1) : -1]
                p = tf.nn.softmax(np.sort(last_pred)[: -(top_k + 1) : -1]).numpy()

                candidate = np.random.choice(top_k, 1, p=p)[0]
                candidate = idx[candidate]
            else:
                raise Exception("top_k must bigger than 0.")

            decoder_ids.append(candidate)

            if candidate == self.tokenizer.eos_token_id:
                break

        return decoder_ids

    def generate_beam(
        self,
        input_text=None,
        input_ids=None,
        decoder_ids=None,
        num_beams=1,
        max_length=128,
        temperature=1.0,
    ):
        input_ids, decoder_ids = self.start_setting(
            input_text=input_text,
            input_ids=input_ids,
            decoder_ids=decoder_ids,
        )

        # input_ids = tf.repeat(input_ids, length, axis=0)
        decoder_ids = np.array([decoder_ids])
        input_embed = None

        if decoder_ids.ndim != 2:
            raise ValueError(
                "ndim of decoder_ids must be 2, but given " + str(decoder_ids.ndim)
            )
        sequences = [[decoder_ids[0].tolist(), 1.0]]

        for _ in range(max_length - 1):
            all_candidates = list()

            sequences_temp = [
                s for s in sequences if s[0][-1] != self.tokenizer.eos_token_id
            ]

            length = len(sequences_temp)

            inputs = {
                "input_ids": input_ids if input_embed is None else None,
                "input_embed": (
                    tf.repeat(input_embed, length, axis=0)
                    if input_embed is not None
                    else None
                ),
                "decoder_input_ids": tf.constant(
                    [s[0] for s in sequences_temp], tf.int32
                ),
            }
            last_pred, input_embed = self.model(inputs)
            last_pred = last_pred[:, -1, :].numpy() / float(temperature)
            last_pred = tf.math.log(tf.math.softmax(last_pred) + 1e-10)

            for i in range(length):
                pred = last_pred[i]
                seq, score = sequences_temp[i]
                new_score = score * -pred

                if seq[-1] == self.tokenizer.eos_token_id:
                    continue

                all_candidates += [
                    [seq + [j], s] for j, s in enumerate(new_score.numpy())
                ]

            args = [s[1] for s in all_candidates]
            idx = np.argsort(args)[:num_beams]
            sequences = [all_candidates[i] for i in idx]
            input_embed = input_embed[:1]

        return sequences[0][0]
