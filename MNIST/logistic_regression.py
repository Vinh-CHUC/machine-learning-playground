from sklearn.linear_model import LogisticRegression
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

lr = LogisticRegression()
lr.fit([x.reshape(-1) for x in x_train], y_train)
lr.score([x.reshape(-1) for x in x_test], y_test)
