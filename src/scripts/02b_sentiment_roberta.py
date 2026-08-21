"""
Phase 1B — CardiffNLP RoBERTa Sentiment Classification
=======================================================
Transformer-based sentiment using cardiffnlp/twitter-roberta-base-sentiment-latest.

Input:
    - data/master_reviews.parquet (corpus_text subset)

Output:
    - data/master_reviews_with_sentiment.parquet (all sentiment columns merged)

See implementation_plan.md §4.2 for full specification.

Hardware: RTX 5090, fp16, dynamic padding, batch size 256.
Expected runtime: ~300K C_text reviews ≈ 15–25 min.
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "master_reviews.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "master_reviews_with_sentiment.parquet"

# Model identifier
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def run_roberta_sentiment(df: pd.DataFrame, batch_size: int = 256) -> pd.DataFrame:
    """
    §4.2 — RoBERTa sentiment (GPU batched)

    TODO:
    - Load tokenizer and model from MODEL_NAME
    - Move model to GPU, set to eval mode, enable fp16
    - Process content_clean in batches of `batch_size`
    - Use dynamic padding (pad to longest in batch, not max_length)
    - Store ALL THREE class probabilities (not just argmax):
        - roberta_neg, roberta_neu, roberta_pos
    - Derive roberta_label from argmax
    - The probability mass is needed for aspect-weighted aggregation in Phase 2
    """
    pass


def add_star_proxy_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    §4.3 — Star-rating proxy sentiment (Method C)

    TODO:
    - 1-2★ → 'Negative'
    - 3★   → 'Neutral'
    - 4-5★ → 'Positive'
    - Store as star_label

    KNOWN LIMITATION: star is app-level verdict, so a 5★ review with
    "only complaint is the qibla is off" is mislabelled at aspect level.
    This divergence is exactly what RQ2 exploits — measure it, don't assume it away.
    """
    pass


def merge_all_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge VADER results (from 02a) with RoBERTa and star-proxy results.

    TODO:
    - Load VADER results (if saved separately)
    - Merge vader_*, roberta_*, star_label columns onto the master DataFrame
    - Save to OUTPUT_PATH
    """
    pass


def main():
    # TODO:
    # df = pd.read_parquet(INPUT_PATH)
    # df_text = df[df['corpus_text']].copy()
    # df_text = run_roberta_sentiment(df_text)
    # df_text = add_star_proxy_sentiment(df_text)
    # df = merge_all_sentiment(df)  # merge VADER + RoBERTa + star
    # df.to_parquet(OUTPUT_PATH, index=False)
    pass


if __name__ == "__main__":
    main()
