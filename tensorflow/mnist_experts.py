# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



# LOW LEVEL UTILITIES

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# LAYER WRAPPERS

def conv2d(x, W):
    """
    x = [batch, in_height, in_width, in_channels]
    filter = [filter_height, filter_width, in_channels, out_channels]
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME") 


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# CONVOLUTIONAL BLOCKS
def conv_block(in_tensor, shape):
    """
    Defines a:
    * Relu(2D Convolutional layer)
    * 2X2 max_pool
    """
    W_conv = weight_variable(shape)
    b_conv = bias_variable(shape[-1:])

    h_conv = tf.nn.relu(conv2d(in_tensor, W_conv) + b_conv)
    h_pool = max_pool_2x2(h_conv)

    return h_pool


# FULLY CONNECTED LAYER
def full_layer(in_tensor, shape, keep_prob, *, relu=False, dropout=False):
    W_fc = weight_variable(shape)
    b_fc = bias_variable(shape[-1:])

    in_tensor_flat = tf.reshape(in_tensor, [-1, shape[0]])

    if relu:
        h_fc = tf.nn.relu(tf.matmul(in_tensor_flat, W_fc) + b_fc)
    else:
        h_fc = tf.matmul(in_tensor_flat, W_fc) + b_fc

    if dropout:
        h_fc_drop = tf.nn.dropout(h_fc, keep_prob)
        return h_fc_drop

    return h_fc


def build_model(x, expected_y, dropout_prob):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # Layers
    conv_1 = conv_block(x_image, [5, 5, 1, 32])
    conv_2 = conv_block(conv_1, [5, 5, 32, 64])
    f_1 = full_layer(conv_2, [7 * 7 * 64, 1024], dropout_prob, relu=True, dropout=True)
    y = full_layer(f_1, [1024, 10], dropout_prob, relu=False, dropout=False)

    # Loss and Optimizer
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=expected_y, logits=y)
    )
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    return train_step, y, cross_entropy

def main(training_examples_count):
    # Placeholders
    x = tf.placeholder(tf.float32, shape=[None, 784])
    expected_y = tf.placeholder(tf.float32, shape=[None, 10])
    dropout_prob = tf.placeholder(tf.float32)

    # Building model
    train_op, predicted_y, loss = build_model(x, expected_y, dropout_prob)

    # Metrics
    correct_prediction = tf.equal(tf.argmax(predicted_y, 1), tf.argmax(expected_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for i in range(training_examples_count):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            training_metrics = {x: batch[0], expected_y: batch[1], dropout_prob: 1}
            test_metrics = {x: mnist.test.images, expected_y: mnist.test.labels, dropout_prob: 1}
            print("step {}, training accuracy {}, loss {}, test accuracy {}".format(
                i,
                accuracy.eval(training_metrics),
                loss.eval(training_metrics),
                accuracy.eval(test_metrics)
            ))
        train_op.run(feed_dict={x: batch[0], expected_y: batch[1], dropout_prob: 0.5})
