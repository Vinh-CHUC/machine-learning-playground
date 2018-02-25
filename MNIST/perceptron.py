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


def build_perceptron_layer(x, in_dim, out_dim):
    # weights
    W = tf.Variable(tf.random_normal([in_dim, out_dim]))
    b = tf.Variable(tf.random_normal([out_dim]))

    # output and expected output
    return tf.add(tf.matmul(x, W), b)


def build_multilayer_perceptron(x):
    a1 = build_perceptron_layer(x, N_PIXELS, 300)
    a2 = build_perceptron_layer(a1, 300, 100)
    out = build_perceptron_layer(a2, 100, N_CLASSES)
    return out


def train(learning_rate=0.001, epochs=15, batch_size = 100):
    # inputs
    X = tf.placeholder(tf.float32, shape=[None, N_PIXELS])

    # outputs
    out = build_multilayer_perceptron(X)

    # expected output
    Y = tf.placeholder(tf.float32, shape=[None, N_CLASSES])

    # loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

    # accuracy
    pred = tf.nn.softmax(out)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            for chunk_start in range(0, len(X_TRAIN), batch_size):
                batch_slice = slice(chunk_start, chunk_start + batch_size)
                batch_x = X_TRAIN[batch_slice]
                batch_y = Y_TRAIN[batch_slice]

                _, l = sess.run(
                    [train_op, loss],
                    feed_dict={X: batch_x, Y: batch_y}
                )

            print("Epoch {}, loss {}".format(epoch, l))
            print("Accuracy:", accuracy.eval({X: X_TEST, Y: Y_TEST}))
