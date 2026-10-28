"""Phase 3: model development – a dense neural network and a convolutional neural network."""
from tensorflow import keras

from . import config


def _compile(model):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_mlp() -> keras.Sequential:
    """Dense network: 28×28 image -> Flatten (784) -> 128 ReLU -> Dropout -> 64 ReLU -> 10 Softmax."""
    h1, h2 = config.MLP_UNITS
    return _compile(keras.Sequential([
        keras.layers.Input(shape=(28, 28)),
        keras.layers.Flatten(name="flatten"),
        keras.layers.Dense(h1, activation="relu", name="hidden_1"),
        keras.layers.Dropout(config.MLP_DROPOUT, name="dropout"),
        keras.layers.Dense(h2, activation="relu", name="hidden_2"),
        keras.layers.Dense(10, activation="softmax", name="output"),
    ], name="mnist_dense_nn"))


def build_cnn() -> keras.Sequential:
    """Convolutional network: two Conv-ReLU-MaxPool blocks, then Dropout and a 10-way Softmax."""
    f1, f2 = config.CNN_FILTERS
    return _compile(keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(f1, (3, 3), activation="relu", name="conv_1"),
        keras.layers.MaxPooling2D((2, 2), name="pool_1"),
        keras.layers.Conv2D(f2, (3, 3), activation="relu", name="conv_2"),
        keras.layers.MaxPooling2D((2, 2), name="pool_2"),
        keras.layers.Flatten(name="flatten"),
        keras.layers.Dropout(config.CNN_DROPOUT, name="dropout"),
        keras.layers.Dense(10, activation="softmax", name="output"),
    ], name="mnist_cnn"))
