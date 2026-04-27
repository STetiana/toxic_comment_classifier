# Toxic Comment Classifier

Toxic language detection is harder than it looks. Sarcasm, coded 
language, and context-dependence make it one of the more genuinely 
difficult text classification problems in NLP — even state-of-the-art 
models struggle with rare categories like `threat` and `identity_hate`.

I benchmarked three approaches of increasing complexity on the 
[Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) 
dataset to see how much difference model complexity actually makes.

---

## Results

| # | Model | Public AUC | Private AUC |
|---|-------|:----------:|:-----------:|
| 🥇 | BERT (`bert-base-uncased`) | **0.931** | **0.928** |
| 🥈 | Bidirectional LSTM | 0.828 | 0.810 |
| 🥉 | LinearSVC + TF-IDF | 0.747 | 0.740 |

BERT wins by a meaningful margin, but the BiLSTM puts up a strong 
result for a model trained from scratch in ~4 minutes per epoch.

---

## The Problem

Each comment can belong to multiple toxicity categories simultaneously 
(multi-label classification across 6 labels):

`toxic` · `severe_toxic` · `obscene` · `threat` · `insult` · `identity_hate`

The dataset is heavily imbalanced — only 9.6% of comments are toxic 
at all, and some categories are extremely rare:

| Label | Count | % of dataset |
|-------|-------|:------------:|
| toxic | 15,294 | 9.58% |
| obscene | 8,449 | 5.29% |
| insult | 7,877 | 4.94% |
| severe_toxic | 1,595 | 1.00% |
| identity_hate | 1,405 | 0.88% |
| threat | 478 | 0.30% |

This imbalance is the central challenge. A model predicting "not toxic" 
for everything would still hit ~90% accuracy.

---

## Models

### 1. BERT — PyTorch + Hugging Face
`bert-base-uncased` fine-tuned with a 6-label classification head.

- Tokenizer: `AutoTokenizer`, max length 128  
- Loss: `BCEWithLogitsLoss` with per-label positive weights  
- Optimizer: AdamW, lr=2e-5, batch size=8  
- 5 epochs (training loss: 0.355 → 0.177)

### 2. Bidirectional LSTM — TensorFlow / Keras
Built from scratch — no pretrained weights, surprisingly competitive.

- Architecture: `TextVectorization` (20k vocab) → `Embedding` (128d) → `BiLSTM` (64) → `Dense` (6, sigmoid)  
- Loss: `binary_crossentropy` with per-sample class weights (range: 1.0–325.6)  
- Early stopped at epoch 3 (val AUC: 0.966)  

### 3. LinearSVC — scikit-learn
The baseline. Fast, interpretable, and more competitive than you'd expect 
for a bag-of-words approach. I compared three classifiers before settling on it:

| Classifier | F1 |
|---|:---:|
| Naive Bayes | 0.22 |
| Logistic Regression | 0.67 |
| **LinearSVC** | **0.72** |

---

## Key Findings

- **BERT's contextual understanding gives it a clear edge** — it learns 
that the same word can be toxic or benign depending on context, something 
TF-IDF fundamentally can't capture.
- **The BiLSTM is surprisingly strong** for a model with no pretrained 
knowledge — sequential context matters even without transformers.
- **`threat` and `identity_hate` were hard for every model** — with only 
478 and 1,405 examples respectively, there isn't enough signal. This is 
a data problem as much as a modelling problem.
- **Accuracy is the wrong metric here** — all models score high on accuracy 
due to class imbalance. AUC and F1 per label tell the real story.