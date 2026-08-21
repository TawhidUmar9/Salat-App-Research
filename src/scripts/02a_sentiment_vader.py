"""
Phase 1A — VADER Sentiment Classification
==========================================
Fast, interpretable lexicon-based sentiment baseline.

Input:
    - data/master_reviews.parquet (corpus_text subset)

Output:
    - Adds VADER columns to the sentiment parquet (merged in 02b or a final merge step)

See implementation_plan.md §4.1 for full specification.
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "master_reviews.parquet"


def run_vader_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    §4.1 — VADER sentiment on content_clean

    TODO:
    - Import nltk.sentiment.vader.SentimentIntensityAnalyzer
    - Download vader_lexicon if not present
    - Run on content_clean column (C_text subset only)
    - Store: vader_neg, vader_neu, vader_pos, vader_compound
    - Derive vader_label:
        - compound >= 0.05  → 'Positive'
        - compound <= -0.05 → 'Negative'
        - else              → 'Neutral'
    """
    pass


def main():
    # TODO:
    # df = pd.read_parquet(INPUT_PATH)
    # df_text = df[df['corpus_text']].copy()
    # df_text = run_vader_sentiment(df_text)
    # Save or merge results
    pass


if __name__ == "__main__":
    main()
