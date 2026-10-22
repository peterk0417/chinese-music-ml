# Dataset

The classification dataset was built from traditional Chinese instrument recordings collected from public online sources.

## Preprocessing

Recordings were:

1. converted to a usable audio format;
2. trimmed to the first 30 seconds;
3. split into 3-second clips;
4. converted into numerical feature vectors.

The recovered large feature table contained approximately **80,000 three-second samples**, **57 numerical audio features**, and **218 classification labels**.

The research also includes detailed analysis of 11 named traditional Chinese instruments:

Erhu, Pipa, Sanxian, Dizi, Suona, Zhuiqin, Zhongruan, Liuqin, Guzheng, Yangqin, and Sheng.

## Feature representation

Each clip is represented with audio features including chroma STFT, RMS, spectral centroid, spectral bandwidth, spectral rolloff, zero-crossing rate, harmonic/percussive statistics, tempo, and MFCC-derived features.

The resulting classification input contains **57 numerical features**.
