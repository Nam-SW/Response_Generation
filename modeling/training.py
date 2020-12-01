from typing import Dict, Tuple, List

import tensorflow as tf

from modeling.model import DialogWithAuxility


def build_model(hparams: Dict):
    mirrored_strategy = tf.distribute.MirroredStrategy()
    # mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

    with mirrored_strategy.scope():
        model = DialogWithAuxility(
            vocab_size=hparams["vocab_size"],
            max_len=hparams["max_len"],
            utterance_size=hparams["utterance_size"],
            embedding_size=hparams["embedding_size"],
            hidden_size=hparams["hidden_size"],
            attention_head=hparams["attention_head"],
            encoder_block=hparams["encoder_block"],
            dropout_rate=hparams["dropout_rate"],
            FFNN_size=hparams["FFNN_size"],
        )
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=hparams["learning_rate"]),
            metrics=["acc"],
        )

        _ = model(
            (
                {
                    "input_contexts": tf.random.uniform(
                        (1, hparams["utterance_size"], hparams["max_len"]),
                        maxval=hparams["vocab_size"],
                        dtype=tf.int32,
                    ),
                    "input_response": tf.random.uniform(
                        (1, hparams["max_len"]),
                        maxval=hparams["vocab_size"],
                        dtype=tf.int32,
                    ),
                }
            )
        )

        model.summary()

    return model


def get_callbacks(
    hparams: Dict, *additional_callback: List[tf.keras.callbacks.Callback]
):
    callback = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=hparams["EarlyStopping_patience"],
            mode="min",
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=hparams["ReduceLROnPlateau_factor"],
            patience=hparams["ReduceLROnPlateau_patience"],
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=hparams["checkpoint_path"],
            save_best_only=True,
            save_weights_only=True,
        ),
    ]

    return callback


def model_training(dataset: Tuple, hparams: Dict, load=True, **training_kwargs):
    context_train, response_train, y_train = dataset
    model = build_model(hparams)
    callback_list = get_callbacks(hparams)

    history = model.fit(
        (context_train, response_train),
        y_train,
        batch_size=hparams["batch_size"],
        epochs=hparams["epochs"],
        validation_split=hparams["validation_split"],
        verbose=hparams["verbose"],
        callbacks=callback_list,
        **training_kwargs
    )

    return model, history


def model_evaluate(dataset: Tuple, model, **kwargs):
    context_test, response_test, y_test = dataset

    print(model.evaluate((context_test, response_test), y_test), **kwargs)
