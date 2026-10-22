# Results

## Classification

| Model | Testing accuracy |
|---|---:|
| Keras Sequential | **99.2%** |
| kNN | **98.6%** |
| CNN + LSTM | **98.1%** |
| Random Forest | **98.0%** |
| CNN + LSTM, deeper model | **97.8%** |
| XGB | **96.6%** |
| Bidirectional LSTM | **96.4%** |
| Conv2D model | **94.1%** |
| Logistic Regression | **93.4%** |
| SVC | **90.8%** |
| LSTM | **82.5%** |
| XGBRF | **68.3%** |
| SGD | **65.7%** |
| k-Means clustering | **33.7%** |

The best-performing classifier was the Keras Sequential model, reported at approximately **99.21% accuracy**.

The surviving kNN experiment preserves an exact testing result of:

```text
0.986098990150854
```

or **98.61%**.

## Music generation

The LSTM generation experiments successfully produced audio from instrument recordings. Increasing the amount of source audio improved the consistency of the generated output.

Generated and original audio were compared using:

- waveform
- STFT
- mel-spectrogram
- MFCC
