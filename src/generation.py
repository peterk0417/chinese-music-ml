from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.io.wavfile import read, write
from tensorflow import keras
from tensorflow.keras import layers


def make_sequences(channel: np.ndarray, look_back: int = 10):
    channel = np.asarray(channel)
    n = len(channel) - look_back - 1
    X = np.array([channel[i:i + look_back] for i in range(n)], dtype=np.float32)
    y = np.array([channel[i + look_back] for i in range(n)], dtype=np.float32)
    return X.reshape((-1, 1, look_back)), y


def build_waveform_lstm(look_back: int = 10) -> keras.Model:
    model = keras.Sequential([
        keras.Input(shape=(1, look_back)),
        layers.LSTM(100, activation="linear"),
        layers.LeakyReLU(),
        layers.Dense(50, activation="linear"),
        layers.LeakyReLU(),
        layers.Dense(25, activation="linear"),
        layers.LeakyReLU(),
        layers.Dense(12, activation="linear"),
        layers.LeakyReLU(),
        layers.Dense(1, activation="linear"),
        layers.LeakyReLU(),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def load_stereo_wav(path: str | Path):
    sr, data = read(path)
    data = np.asarray(data)
    if data.ndim == 1:
        data = np.column_stack([data, data])
    return sr, data[:, :2]


def write_stereo_wav(path: str | Path, sr: int, left: np.ndarray, right: np.ndarray):
    stereo = np.column_stack([left, right])
    stereo = np.clip(stereo, -32768, 32767).astype(np.int16)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write(path, sr, stereo)
