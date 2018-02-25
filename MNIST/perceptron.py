from keras.datasets import mnist
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = mnist.load_data()

n_input = x_train.shape[1] * x_train.shape[2]

lr = LogisticRegression(verbose=True)
lr.fit([x.reshape(-1) for x in x_train], y_train)
lr.score([x.reshape(-1) for x in x_test], y_test)
