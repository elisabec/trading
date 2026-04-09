# Twitter Engagement Prediction — Deep Learning on Alternative Data

Predicting tweet engagement (high vs. low retweets) using deep learning models trained on text content and metadata features. Built as the final project for MIT Sloan's 15.S04 Hands-on Deep Learning (Spring 2022).

---

## Motivation

Alternative data — social media, news, satellite imagery — is increasingly central to quantitative research. This project applies NLP and deep learning to predict which tweets will generate above-median engagement, demonstrating a text-to-quantitative-signal pipeline applicable to financial sentiment analysis and social media alpha.

---

## Data

- **Source:** Twitter API — historical tweets from U.S. politicians (e.g. AOC)
- **Scope:** ~15k original tweets (no retweets), chronological 80/20 train/test split
- **Target variable:** Binary — "High" or "Low" engagement relative to the user's median retweet count
- **Features:**
  - **Text:** Raw tweet content (cleaned, tokenized)
  - **Metadata:** Weekday, hour of day, quote status, reply status

---

## Methodology — 6 Model Comparison

Systematic evaluation of increasingly sophisticated architectures:

| # | Model | Text Representation | Test Accuracy |
|---|-------|-------------------|---------------|
| 1 | Baseline Dense | Unigram multi-hot | 59.95% |
| 2 | GloVe Embedding | Pre-trained GloVe 100d | 54.64% |
| 3 | Trained Embedding | Custom 64d | 58.96% |
| 4 | BERT Embedding | Pre-trained BERT (768d) | 59.93% |
| 5 | Bigram Dense | Bigram multi-hot | 61.98% |
| 6 | BERT + Metadata | BERT + weekday/hour/quote/reply | **64.57%** |

All models use categorical cross-entropy loss, Adam optimizer, and 30 training epochs.

---

## Key Results

- The concatenated BERT + Metadata model achieves the best performance at **64.57%** test accuracy
- Adding metadata (timing, reply context) to text features provides a ~5pp lift over BERT alone
- Bigram representations outperform unigram baselines, confirming that word-pair context carries signal
- GloVe underperforms simpler approaches on short tweet text, likely due to domain mismatch

---

## Quick start

→ Start here: [`notebooks/model_comparison.ipynb`](notebooks/model_comparison.ipynb)

---

## Tech Stack

`Python` · `TensorFl