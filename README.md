# Chinese Musical Instrument Classification and Music Generation

Code accompanying the research paper:

**Music Generation and Classification from Chinese Musical Instruments Using Deep Learning**  
Joonho Kwak, IYRC Journal, 2024  
DOI: https://doi.org/10.34614/JIYRC202418  
Paper: https://journal.the-iyrc.org/jiyrc18-joonho-kwak.html

This project applies machine learning and deep learning to traditional Chinese music for two tasks:

- musical instrument classification
- LSTM-based music generation

Audio recordings were standardized and split into 3-second clips. Numerical audio features were extracted and used to compare classical machine-learning models with deep-learning architectures.

## Classification results

The paper reports the following testing accuracies:

| Model | Accuracy |
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

The best model was the Keras Sequential classifier, reported in the paper at approximately **99.21% accuracy**.

The surviving classification notebook also preserves a kNN result of:

```text
0.986098990150854
```

which corresponds to **98.61% testing accuracy** and matches the paper's reported kNN result.

## Dataset

The paper's classification dataset was constructed from thousands of traditional Chinese instrument recordings collected from public online sources.

The preprocessing pipeline:

1. converts recordings to WAV/MP3 where necessary;
2. keeps the first 30 seconds of each recording;
3. splits the recordings into 3-second clips;
4. extracts numerical audio features;
5. applies Min-Max scaling;
6. uses an 80/20 train-test split.

The recovered large feature table contained:

```text
80,207 samples
57 numerical audio features
218 classification labels
```

The paper also contains focused analysis of 11 traditional Chinese instruments:

- Erhu
- Pipa
- Sanxian
- Dizi
- Suona
- Zhuiqin
- Zhongruan
- Liuqin
- Guzheng
- Yangqin
- Sheng

## Audio features

The classification pipeline uses features including:

- chroma STFT
- RMS energy
- spectral centroid
- spectral bandwidth
- spectral rolloff
- zero-crossing rate
- harmonic/percussive statistics
- tempo
- MFCC features

These produce a 57-dimensional numerical representation for each audio clip.

## Models

### Classical machine learning

The project evaluates:

- k-Nearest Neighbors
- Logistic Regression
- SGD
- XGBoost
- Random Forest
- Support Vector Classifier
- XGBoost Random Forest

The strongest classical model was kNN, reaching approximately **98.6% testing accuracy**.

### Deep learning

The project evaluates:

- Keras Sequential feedforward network
- LSTM
- Bidirectional LSTM
- CNN + LSTM
- deeper CNN + Bidirectional LSTM
- Conv2D classifier

The Keras Sequential model achieved the highest reported accuracy at approximately **99.21%**.

## Music generation

An LSTM-based model was also developed to generate audio based on traditional Chinese instrument recordings.

The generation experiments include:

- training from a single instrument recording;
- training from multiple recordings of the same instrument;
- waveform generation for stereo audio;
- comparison of generated and original audio using waveform, STFT, mel-spectrogram, and MFCC visualizations.

## Repository structure

```text
.
├── README.md
├── RESULTS.md
├── requirements.txt
├── CITATION.cff
├── data/
│   ├── README.md
│   └── recovered_small/
├── notebooks/
│   ├── 01_data_preparation_and_features.ipynb
│   ├── 02_classical_ml.ipynb
│   ├── 03_deep_learning.ipynb
│   ├── 04_sequential_classifier.ipynb
│   ├── 05_lstm_music_generation.ipynb
│   ├── 06_generated_music_analysis.ipynb
│   ├── 07_clustering.ipynb
│   └── 08_conditional_generation_experiment.ipynb
├── src/
│   ├── features.py
│   ├── labels.py
│   ├── classical.py
│   ├── deep_models.py
│   └── generation.py
└── docs/
    ├── DATASET_NOTES.md
    └── paper_results.csv
```

## Reproducing the experiments

Install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The original large `new_features_3sec.csv` file is not included in this repository. If the original audio dataset is available, notebook 01 can be used to regenerate the feature table.

Then run the notebooks in order.

## Note on the 99.21% model

The exact saved notebook/model checkpoint that produced the paper's **99.21% Sequential result** is no longer available.

`04_sequential_classifier.ipynb` reconstructs the Sequential architecture and training setup from the surviving research code. It is provided so the model structure and experiment can still be reproduced, but it does not claim that a newly rerun model is the original saved 99.21% run.

## Publication

Kwak, J. (2024). *Music Generation and Classification from Chinese Musical Instruments Using Deep Learning*. IYRC Journal.  
DOI: **10.34614/JIYRC202418**
