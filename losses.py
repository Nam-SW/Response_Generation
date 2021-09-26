import tensorflow as tf


def mle_loss(y, pred):
    mask = tf.math.logical_not(tf.math.equal(y, 0))

    reshaped_y = y[:, :, tf.newaxis]
    batch = tf.broadcast_to(
        tf.range(y.shape[0], dtype=y.dtype)[:, tf.newaxis, tf.newaxis],
        reshaped_y.shape,
    )
    vocab = tf.broadcast_to(
        tf.range(y.shape[1], dtype=y.dtype)[tf.newaxis, :, tf.newaxis],
        reshaped_y.shape,
    )

    reshaped_y = tf.concat([batch, vocab, reshaped_y], axis=2)
    pred = tf.gather_nd(tf.nn.softmax(pred, axis=-1), reshaped_y)

    loss = tf.math.log(pred)
    mask = tf.cast(mask, dtype=loss.dtype)

    loss = -(tf.math.reduce_sum(loss * mask, axis=-1))

    return tf.math.reduce_sum(loss) / tf.math.reduce_sum(mask)
