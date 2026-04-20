# Toxic Language Classifier

Multi-label toxic comment classifier built with three approaches: BERT (PyTorch), LinearSVC (scikit-learn), and Bidirectional LSTM (TensorFlow).

## Problem

Automatically detect toxic comments across 6 categories simultaneously. Each comment can belong to multiple categories at once (multi-label classification).

**Labels:** `toxic` · `severe_toxic` · `obscene` · `threat` · `insult` · `identity_hate`

**Evaluation metric:** Mean column-wise ROC AUC (average of individual AUCs per label)

## Dataset

| Property | Value |
|----------|-------|
| Total comments | 159,571 |
| Missing values | None |
| Duplicate entries | None |
| Class balance | Imbalanced — ~9.6% toxic |

| Label | Count | % |
|-------|-------|---|
| toxic | 15,294 | 9.58% |
| obscene | 8,449 | 5.29% |
| insult | 7,877 | 4.94% |
| severe_toxic | 1,595 | 1.00% |
| identity_hate | 1,405 | 0.88% |
| threat | 478 | 0.30% |

## Kaggle Results (Mean Column-Wise ROC AUC)

| # | Model | Public Score | Private Score |
|---|-------|:------------:|:-------------:|
| 🥇 | **BERT** (PyTorch) | **0.93062** | **0.92755** |
| 🥈 | **BiLSTM** (TensorFlow) | **0.82839** | **0.80964** |
| 🥉 | **LinearSVC** (scikit-learn) | **0.74718** | **0.74032** |

## Models

### 1. BERT (PyTorch) — best model
- `bert-base-uncased` fine-tuned with a 6-label classification head
- Tokenizer: `AutoTokenizer`, max length 128
- Loss: `BCEWithLogitsLoss` with per-label positive weights
- Optimizer: AdamW, lr=2e-5, batch size=8
- Split: 70/20/10 — trained for 5 epochs (loss: 0.355 → 0.177)

### 2. Bidirectional LSTM (TensorFlow)
- `TextVectorization(20k vocab, 200 tokens)` → `Embedding(128d)` → `Bi-LSTM(64)` → `Dense(6, sigmoid)`
- Loss: `binary_crossentropy` with per-sample class weights (range: 1.0–325.6)
- Optimizer: Adam, batch size=64
- Split: 70/15/15 — early stopped at epoch 3 (val AUC: 0.966)
- GPU: NVIDIA GeForce RTX 3060 Laptop

### 3. LinearSVC (scikit-learn)
- Pipeline: `TfidfVectorizer` → `OneVsRestClassifier(LinearSVC)`
- Split: 80/20
- Three classifiers compared: Naive Bayes (F1=0.22), Logistic Regression (F1=0.67), **LinearSVC (F1=0.72)**

## Project Structure

```
toxic_classifier/
├── toxic_language_classifier.ipynb   # main notebook
├── train.csv                         # training data
├── test.csv                          # test data
├── sample_submission.csv             # submission format
├── submission_bert.csv               # BERT predictions  (ROC AUC: 0.931)
├── submission_tf.csv                 # TF BiLSTM predictions (ROC AUC: 0.828)
├── submission_SVC.csv                # SVC predictions   (ROC AUC: 0.747)
├── model.safetensors                 # saved BERT weights
├── config.json                       # BERT model config
├── tokenizer.json                    # BERT tokenizer
├── vocab.txt                         # BERT vocabulary
└── tokenizer_config.json
```

## Setup

```bash
pip install tensorflow[and-cuda] pandas matplotlib seaborn scikit-learn wordcloud
pip install torch transformers
```

## Key Findings

1. **BERT achieved the best Kaggle score (public ROC AUC 0.931)** — deep contextual understanding gives it a strong advantage over simpler models
2. **TF BiLSTM ranked second (0.828)** — strong result for a model trained in ~4 minutes per epoch with no pretrained weights
3. **LinearSVC ranked third (0.747)** — fastest to train but TF-IDF lacks the semantic depth of neural approaches
4. **`threat` and `identity_hate`** are the hardest labels across all models — fewest samples (478 and 1,405) and lowest F1 in every approach
5. **Class imbalance** is the central challenge — all models used weighting strategies, yet rare labels remain difficult
