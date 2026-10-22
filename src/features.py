from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import pandas as pd


def extract_57_features(file_path: str | Path) -> dict[str, float]:
    """Feature extractor recovered from the original notebooks."""
    y, sr = librosa.load(file_path, sr=44100)

    chroma_stft = librosa.feature.chroma_stft(y=y)
    rms = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y)
    rolloff = librosa.feature.spectral_rolloff(y=y)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    harmonic = librosa.effects.harmonic(y=y)
    percussive = librosa.effects.percussive(y=y)
    tempo, _ = librosa.beat.beat_track(y=y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    features = {
        "chroma_stft_mean": float(chroma_stft.mean()),
        "chroma_stft_var": float(chroma_stft.var()),
        "rms_mean": float(rms.mean()),
        "rms_var": float(rms.var()),
        "spectral_centroid_mean": float(spectral_centroid.mean()),
        "spectral_centroid_var": float(spectral_centroid.var()),
        "spectral_bandwidth_mean": float(spectral_bandwidth.mean()),
        "spectral_bandwidth_var": float(spectral_bandwidth.var()),
        "rolloff_mean": float(rolloff.mean()),
        "rolloff_var": float(rolloff.var()),
        "zero_crossing_rate_mean": float(zero_crossing_rate.mean()),
        "zero_crossing_rate_var": float(zero_crossing_rate.var()),
        "harmony_mean": float(harmonic.mean()),
        "harmony_var": float(harmonic.var()),
        "perceptr_mean": float(percussive.mean()),
        "perceptr_var": float(percussive.var()),
        "tempo": float(np.asarray(tempo).squeeze()),
    }

    for i in range(20):
        features[f"mfcc_mean_{i + 1}"] = float(mfcc[i].mean())
    for i in range(20):
        features[f"mfcc_var_{i + 1}"] = float(mfcc[i].var())

    return features


def feature_row(file_path: str | Path) -> dict:
    path = Path(file_path)
    y, sr = librosa.load(path, sr=44100)
    return {
        "filename": path.name,
        "length": float(librosa.get_duration(y=y, sr=sr)),
        **extract_57_features(path),
    }


def feature_table(audio_paths: Iterable[str | Path]) -> pd.DataFrame:
    return pd.DataFrame([feature_row(p) for p in audio_paths])


def feature_table_from_directory(audio_dir: str | Path) -> pd.DataFrame:
    audio_dir = Path(audio_dir)
    paths = sorted(
        [*audio_dir.glob("*.wav"), *audio_dir.glob("*.mp3")],
        key=lambda p: p.name,
    )
    if not paths:
        raise FileNotFoundError(f"No WAV/MP3 files found in {audio_dir}")
    return feature_table(paths)
