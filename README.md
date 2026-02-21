# Travel Destination Classification

Multi-class text classification: predict travel destination countries from descriptions.

## Overview

| Item | Value |
|------|-------|
| Dataset | 1,011 samples, 54 countries |
| Feature | TF-IDF (1000 features, 1-2 grams) |
| Best Model | Logistic Regression (66.49%) |
| Language | Python 3.7+ |

## Results

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| k-NN | 36.76% | 0.406 |
| Logistic Regression | **66.49%** | **0.611** |
| Random Forest | 58.38% | 0.545 |

## Installation

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

## Quick Start

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# TF-IDF Features
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
X = vectorizer.fit_transform(descriptions)

# Train
model = LogisticRegression(C=10.0)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.4f}")
```

## Documentation

- **Project_Report.pdf** - Full analysis and results
- **01_EDA.ipynb** - Exploratory data analysis
- **02_preprocessing.ipynb** - Data cleaning
- **03_modeling.ipynb** - Model training & evaluation

## Key Insights

✅ Country-specific keywords are strong predictors  
✅ Logistic Regression outperforms ensemble methods  
✅ Hierarchical classification improves interpretability  
⚠️ Class imbalance affects minority classes  
⚠️ Long descriptions slightly reduce accuracy  

## Top Discriminative Words

| Country | Top Words |
|---------|-----------|
| Italy | venice, rome, canal, colosseum |
| France | eiffel, paris, tower, foreground |
| Spain | barcelona, sagrada, alhambra, madrid |
| Japan | japan, temple, tokyo, japanese |

## Error Analysis

**Easiest to classify:** Spain, Switzerland, Palestine (100%)  
**Hardest to classify:** USA (40%)  

**Top misclassifications:**
- USA → Switzerland
- Turkey → Italy
- USA → Italy

## Limitations

- Class imbalance (rare countries have few samples)
- TF-IDF ignores semantic meaning
- Textual similarity between geographically close countries
- Missing values in auxiliary columns

## Future Improvements

- Word embeddings (Word2Vec, FastText)
- Transformer models (BERT, RoBERTa)
- Class weighting / SMOTE resampling
- Multi-modal approach (text + images)

## Authors

**Sara Ewaida** (1203048)  
**Yara Obaid** (1212482)

Electrical & Computer Engineering Department  
Birzeit University | ENCS5341 | January 2026

---

For more details, see Project_Report.pdf and the Jupyter notebooks.
