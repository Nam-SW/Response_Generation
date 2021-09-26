import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import reverse

# beam search


class Generator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def predict(
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
            input_ids = self.tokenizer.encode(
                input_text,
                return_tensors="tf",
            )
        elif input_ids is not None:
            input_ids = tf.constant(input_ids, dtype=tf.int32)
            input_ids = input_ids if input_ids.ndim == 2 else input_ids[tf.newaxis, :]

        if decoder_ids is None:
            decoder_ids = [self.tokenizer.bos_token_id]

        if isinstance(decoder_ids, np.ndarray):
            decoder_ids = decoder_ids.tolist()
    
    def generate_greedy(
        self,
        input_text=None,
        input_ids=None,
        decoder_ids=None,
        max_length=128,
        temperature=1.0,
    ):
        assert (
            input_ids is not None or input_text is not None
        ), "input_text or input_ids must be required."
        assert (
            input_ids is None or input_text is None
        ), "Only one of input_text and input_ids is required."

        if input_text is not None:
            input_ids = self.tokenizer.encode(
                input_text,
                return_tensors="tf",
            )
        elif input_ids is not None:
            input_ids = tf.constant(input_ids, dtype=tf.int32)
            input_ids = input_ids if input_ids.ndim == 2 else input_ids[tf.newaxis, :]

        if decoder_ids is None:
            decoder_ids = [self.tokenizer.bos_token_id]

        if isinstance(decoder_ids, np.ndarray):
            decoder_ids = decoder_ids.tolist()

        for _ in range(max_length - 1):
            last_pred = self.model(
                input_ids=input_ids,
                decoder_input_ids=tf.constant([decoder_ids], tf.int32),
                temperature=temperature,
            ).numpy()
            # last_pred = tf.nn.softmax(last_pred).numpy()
            last_pred = last_pred[0, -1, :] / float(temperature)
            candidate = int(np.argmax(last_pred, axis=-1))
            decoder_ids.append(candidate)

            if candidate == self.tokenizer.eos_token_id:
                break

        return decoder_ids

    def generate_random_sampling(
        self,
        input_text=None,
        input_ids=None,
        decoder_ids=None,
        max_length=128,
        temperature=1.0,
        top_k=1,
    ):
        assert (
            input_ids is not None or input_text is not None
        ), "input_text or input_ids must be required."
        assert (
            input_ids is None or input_text is None
        ), "Only one of input_text and input_ids is required."

        if input_text is not None:
            input_ids = self.tokenizer.encode(
                input_text,
                return_tensors="tf",
            )
        elif input_ids is not None:
            input_ids = tf.constant(input_ids, dtype=tf.int32)
            input_ids = input_ids if input_ids.ndim == 2 else input_ids[tf.newaxis, :]

        if decoder_ids is None:
            decoder_ids = [self.tokenizer.bos_token_id]

        if isinstance(decoder_ids, np.ndarray):
            decoder_ids = decoder_ids.tolist()

        for _ in range(max_length - 1):
            last_pred = self.model(
                input_ids=input_ids,
                decoder_input_ids=tf.constant([decoder_ids], tf.int32),
                temperature=temperature,
            ).numpy()
            # last_pred = tf.nn.softmax(last_pred).numpy()
            last_pred = last_pred[0, -1, :] / float(temperature)

            if top_k == 1:
                candidate = int(np.argmax(last_pred, axis=-1))
            elif top_k >= 2:
                idx = last_pred.argsort()[:-(top_k + 1):-1]
                p = tf.nn.softmax(np.sort(last_pred)[:-(top_k + 1):-1]).numpy()

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
        assert (
            input_ids is not None or input_text is not None
        ), "input_text or input_ids must be required."
        assert (
            input_ids is None or input_text is None
        ), "Only one of input_text and input_ids is required."

        if input_text is not None:
            input_ids = self.tokenizer.encode(
                input_text,
                return_tensors="tf",
            )
        elif input_ids is not None:
            input_ids = tf.constant(input_ids, dtype=tf.int32)
            input_ids = input_ids if input_ids.ndim == 2 else input_ids[tf.newaxis, :]

        if decoder_ids is None:
            decoder_ids = [[self.tokenizer.bos_token_id]]

        if isinstance(decoder_ids, list):
            decoder_ids = np.array(decoder_ids)

        if decoder_ids.ndim != 2:
            decoder_ids = decoder_ids.reshape((1, -1))
        sequences = [[decoder_ids[0].tolist(), 1.0]]

        for _ in range(max_length - 1):
            all_candidates = list()

            sequences_temp = [
                s for s in sequences if s[0][-1] != self.tokenizer.eos_token_id
            ]

            length = len(sequences_temp)
#             print(tf.constant([s[0] for s in sequences_temp], tf.int32))
            dec_i = tf.constant([s[0] for s in sequences_temp], tf.int32)
            
            try:
                last_pred = self.model(
                    input_ids=tf.repeat(input_ids, length, axis=0),
                    decoder_input_ids=dec_i,
                    temperature=temperature,
                )
            except:
                print(dec_i)
                break
            last_pred = tf.math.log(tf.math.softmax(last_pred[:, -1, :]) + 1e-10)

            for i in range(length):
                pred = last_pred[i]
                seq, score = sequences_temp[i]
                new_score = score * -pred

                if seq[-1] == self.tokenizer.eos_token_id:
                    continue

                all_candidates += [
                    # TODO: top-n sampling
                    [seq + [j], s]
                    for j, s in enumerate(new_score.numpy())
                ]

            args = [s[1] for s in all_candidates]
            idx = np.argsort(args)[:num_beams]
            sequences = [all_candidates[i] for i in idx]

        return sequences[0][0]
