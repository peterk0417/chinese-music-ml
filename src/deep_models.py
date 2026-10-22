from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_reconstructed_sequential(n_features: int, n_classes: int) -> keras.Model:
    """Reconstruct the dense architecture preserved in the lost-run branch.

    The surviving historical cell ends with Dense(10), which conflicts with the
    218-label large pipeline. The reconstruction therefore uses n_classes.
    """
    model = keras.Sequential([
        keras.Input(shape=(n_features,)),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_lstm(n_features: int, n_classes: int) -> keras.Model:
    model = keras.Sequential([
        keras.Input(shape=(1, n_features)),
        layers.LSTM(64),
        layers.Dropout(0.5),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_bidirectional_lstm(n_features: int, n_classes: int) -> keras.Model:
    model = keras.Sequential([
        keras.Input(shape=(n_features, 1)),
        layers.Bidirectional(layers.LSTM(128)),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_lstm(n_features: int, n_classes: int) -> keras.Model:
    model = keras.Sequential([
        keras.Input(shape=(n_features, 1)),
        layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2, padding="same"),
        layers.LSTM(128),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_deep_cnn_bilstm(n_features: int, n_classes: int) -> keras.Model:
    model = keras.Sequential([
        keras.Input(shape=(n_features, 1)),
        layers.Conv1D(128, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv1D(128, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2, padding="same"),
        layers.Dropout(0.3),

        layers.Conv1D(256, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv1D(256, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2, padding="same"),
        layers.Dropout(0.3),

        layers.Bidirectional(layers.LSTM(256, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(128)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_conv2d(n_features: int, n_classes: int) -> keras.Model:
    model = keras.Sequential([
        keras.Input(shape=(n_features, 1, 1)),
        layers.Conv2D(24, (6, 1), activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Conv2D(24, (6, 1), activation=layers.LeakyReLU(negative_slope=0.2)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Conv2D(48, (5, 1), strides=(2, 1), activation=layers.LeakyReLU(negative_slope=0.2)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Conv2D(48, (5, 1), strides=(2, 1), activation=layers.LeakyReLU(negative_slope=0.2)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Conv2D(64, (4, 1), activation=layers.LeakyReLU(negative_slope=0.2)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(200, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(50, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
