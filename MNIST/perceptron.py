import os

from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf


(X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = mnist.load_data()
X_TRAIN = X_TRAIN.reshape(X_TRAIN.shape[0], -1)
X_TEST = X_TEST.reshape(X_TEST.shape[0], -1)
Y_TRAIN = to_categorical(Y_TRAIN, 10)
Y_TEST = to_categorical(Y_TEST, 10)

N_PIXELS = 784
N_CLASSES = 10


def build_perceptron_layer(x, in_dim, out_dim, name="perceptron"):
    with tf.name_scope(name):
        # weights
        W = tf.Variable(tf.random_normal([in_dim, out_dim]), name="W")
        b = tf.Variable(tf.random_normal([out_dim]), name="b")

        tf.summary.histogram(name + "_weights", W)
        tf.summary.histogram(name + "_biases", b)

        # output
        out = tf.nn.sigmoid(tf.add(tf.matmul(x, W), b), name="sigmoid")
        tf.summary.histogram(name + "_out", out)
        return out


def build_multilayer_perceptron(x, name="multilayer_perceptron"):
    with tf.name_scope(name):
        a1 = build_perceptron_layer(x, N_PIXELS, 300, name="perceptron_1")
        a2 = build_perceptron_layer(a1, 300, 100, name="perceptron_2")
        out = build_perceptron_layer(a2, 100, N_CLASSES, name="perceptron_3")
        return out


def get_log_dirname():
    LOG_DIR_ROOT = "/tmp/mnist_demo/"
    return LOG_DIR_ROOT + str(
        (
            sorted([int(x) for x in os.listdir(LOG_DIR_ROOT)])[-1] + 1
        )
    )


def train(learning_rate=0.001, epochs=15, batch_size = 100):
    tf.reset_default_graph()
    # inputs
    X = tf.placeholder(tf.float32, shape=[None, N_PIXELS], name="X")

    # outputs
    out = build_multilayer_perceptron(X, name="multilayer_perceptron")
    pred = tf.nn.softmax(out, name="output")  # Apply softmax to logits

    # expected output
    Y = tf.placeholder(tf.float32, shape=[None, N_CLASSES], name="expected_output")

    # loss
    with tf.name_scope("xent"):
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=Y, name="total_loss"),
            name="loss"
        )
        tf.summary.scalar("loss", loss)
    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss)

    # accuracy
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar("accuraccy", accuracy)

    with tf.Session() as sess:
        merged_summary = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(get_log_dirname())
        writer.add_graph(sess.graph)

        for epoch in range(epochs):
            for chunk_start in range(0, len(X_TRAIN), batch_size):
                batch_slice = slice(chunk_start, chunk_start + batch_size)
                batch_x = X_TRAIN[batch_slice]
                batch_y = Y_TRAIN[batch_slice]

                _, l, s = sess.run(
                    [train_op, loss, merged_summary],
                    feed_dict={X: batch_x, Y: batch_y}
                )
                writer.add_summary(s, epoch)

            print("Epoch {}, loss {}".format(epoch, l))
            print("Accuracy:", accuracy.eval({X: X_TEST, Y: Y_TEST}))
