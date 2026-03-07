import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from utils.data_loader import load_training_data


def build_categorizer():
    train_df = load_training_data()
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=1000)),
        ('clf',   LogisticRegression(max_iter=1000, random_state=42, C=5))
    ])
    pipeline.fit(train_df['description'], train_df['category'])
    print("Categorization model ready")
    return pipeline, len(train_df)


_categorizer, TRAINING_ROWS = build_categorizer()


def categorize_transactions(df):
    descriptions              = df['description'].str.lower().fillna('unknown')
    df['category']            = _categorizer.predict(descriptions)
    probs                     = _categorizer.predict_proba(descriptions)
    df['category_confidence'] = (np.max(probs, axis=1) * 100).round(1)
    return df
