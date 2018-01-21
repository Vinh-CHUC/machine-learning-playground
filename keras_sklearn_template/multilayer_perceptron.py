from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn import model_selection
import numpy


def create_model(optimizer="rmsprop", init="glorot_uniform"):
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


def build_param_grid():
    optimizers = ["rmsprop", "adam"]
    inits = ["glorot_uniform", "normal", "uniform"]
    epochs = [50, 100, 150]
    batches = [5, 10, 20]
    return dict(
        optimizer=optimizers,
        init=inits,
        epochs=epochs,
        batch_size=batches
    )


def get_data():
    seed = 7
    numpy.random.seed(seed)
    dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
    X = dataset[:, 0:8]
    Y = dataset[:, 8]
    return X, Y

def main():
    X, Y = get_data()
    model = KerasClassifier(build_fn=create_model)
    grid = model_selection.GridSearchCV(estimator=model, param_grid=build_param_grid())
    grid_result = grid.fit(X, Y)
