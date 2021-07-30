import tensorflow as tf

def focal_loss(y_true, y_pred):
    gamma = 1.5
    y_true = tf.argmax(y_true, axis=-1)

    probs = tf.nn.softmax(y_pred, axis=-1)

    xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y_true, # batch , w, h (-1 차원 )
        logits=y_pred, # batch, w, h, classes
    )

    y_true_rank = y_true.shape.rank
    probs = tf.gather(probs, y_true, axis=-1, batch_dims=y_true_rank)

    focal_modulation = (1 - probs) ** gamma
    fl_loss = focal_modulation * xent_loss

    # loss = tf.nn.compute_average_loss(fl_loss)
    loss = tf.reduce_mean(fl_loss)

    return loss