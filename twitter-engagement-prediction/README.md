# Twitter Engagement Prediction — Deep Learning on Alternative Data

Predicting tweet engagement (high vs. low retweets) using deep learning models trained on text content and metadata features. Built as the final project for MIT Sloan's 15.S04 Hands-on Deep Learning (Spring 2022).

---

## Motivation

Alternative data — social media, news, satellite imagery — is increasingly central to quantitative research. This project applies NLP and deep learning to predict which tweets will generate above-median engagement, demonstrating a text-to-quantitative-signal pipeline applicable to financial sentiment analysis and social media alpha.

---

## Data

- **Source:** Twitter API — historical tweets from U.S. politicians (e.g. AOC)
- **Scope:** Original tweets only (no retweets)
- **Target variable:** Binary — "High" or "Low" engagement relative to the user's median retweet count
- **Features:**
  - **Text:** Raw tweet content (cleaned, tokenized)
  - **Metadata:** Weekday, hour of day, quote status, reply status

---

## Methodology — 6 Model Comparison

Systematic evaluation of increasingly sophisticated architectures:

| # | Model | Text Representation | Notes |
|---|-------|-------------------|-------|
| 1 | Baseline Dense | Unigram multi-hot | Simplest bag-of-words |
| 2 | GloVe Embedding | Pre-trained GloVe 100d | Transfer learning from general corpus |
| 3 | Trained Embedding | Custom 64d embedding | Learned from tweet data |
| 4 | BERT Embedding | Pre-trained BERT (768d) | Contextual representations |
| 5 | Bigram Dense | Bigram multi-hot | Captures word pairs |
| 6 | BERT + Metadata | BERT + weekday/hour/quote/reply | Concatenated multi-input model |

All models use categorical cross-entropy loss, Adam optimizer, and are evaluated on a chronological train/test split (80/20).

---

## Key Results

- BERT embeddings outperform simpler representations for tweet classification
- Adding metadata (timing, reply context) to text features improves prediction
- Engagement varies significantly by hour of day and weekday — temporal features carry signal

---

## Tech Stack

`Python` · `TensorFlow/Keras` · `BERT (TF Hub)` · `GloVe` · `Twitter API` · `scikit-learn` · `pandas`

---

## Repository Structure
twitter-engagement-prediction/
├── notebooks/
│   └── model_comparison.ipynb    # Full analysis notebook
├── src/
│   ├── data_loader.py            # Tweet retrieval and preprocessing
│   ├── features.py               # Text vectorization, BERT features
│   └── models.py                 # Model architectures (1-6)
├── results/
│   └── model_comparison.csv      # Accuracy results across models
├── README.md
├── requirements.txt
└── .gitignore

---

## Setup
```bash
pip install -r requirements.txt
```

Note: Twitter API credentials are required for data retrieval and are excluded from this repository.