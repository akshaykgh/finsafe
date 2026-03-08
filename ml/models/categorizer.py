import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from utils.data_loader import load_training_data

_categorizer  = None
_training_rows = 0


def _build_categorizer():
    train_df = load_training_data()
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=2000)),
        ('clf',   LogisticRegression(max_iter=1000, random_state=42, C=5))
    ])
    pipeline.fit(train_df['description'], train_df['category'])
    print(f"Categorization model ready ({len(train_df)} training rows)")
    return pipeline, len(train_df)


def get_categorizer():
    global _categorizer, _training_rows
    if _categorizer is None:
        _categorizer, _training_rows = _build_categorizer()
    return _categorizer


def get_training_rows():
    get_categorizer()
    return _training_rows


def categorize_transactions(df):
    model        = get_categorizer()
    descriptions = df['description'].str.lower().fillna('unknown')

    df['category']            = model.predict(descriptions)
    probs                     = model.predict_proba(descriptions)
    df['category_confidence'] = (np.max(probs, axis=1) * 100).round(1)
    return df
