"""
Phase 1A — VADER Sentiment Classification
==========================================
Fast, interpretable lexicon-based sentiment baseline.

Input:
    - data/master_reviews.parquet (corpus_text subset)

Output:
    - data/_checkpoints/vader/vader.parquet  (merged into the sentiment frame by 02b)

See implementation_plan.md §4.1 for full specification.

Usage:
    python src/scripts/02a_sentiment_vader.py
    python src/scripts/02a_sentiment_vader.py --sample 20000
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CKPT_DIR, MASTER_PATH,
    base_parser, log, maybe_sample, require, section, set_seed,
)

OUTPUT_PATH = CKPT_DIR / "vader" / "vader.parquet"

#: Standard VADER thresholds (Hutto & Gilbert 2014).
POS_THRESHOLD = 0.05
NEG_THRESHOLD = -0.05


def _ensure_lexicon() -> None:
    import nltk
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        log("Downloading NLTK vader_lexicon (one time)...")
        nltk.download("vader_lexicon", quiet=True)


def run_vader_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    §4.1 — VADER sentiment on content_clean.

    VADER is English-tuned, so scores for non-English rows are not meaningful.
    We compute them anyway (cheap, and the cross-method comparison in §4.4
    needs aligned rows) but flag them via `vader_applicable` so downstream
    analysis can restrict to C_en.
    """
    section("§4.1  VADER sentiment")
    _ensure_lexicon()

    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from tqdm.auto import tqdm

    sia = SentimentIntensityAnalyzer()
    texts = df["content_clean"].fillna("").tolist()

    scores = [
        sia.polarity_scores(t)
        for t in tqdm(texts, desc="vader", unit="review", file=sys.stderr)
    ]
    s = pd.DataFrame(scores, index=df.index)

    df["vader_neg"] = s["neg"].astype("float32")
    df["vader_neu"] = s["neu"].astype("float32")
    df["vader_pos"] = s["pos"].astype("float32")
    df["vader_compound"] = s["compound"].astype("float32")

    df["vader_label"] = np.select(
        [df["vader_compound"] >= POS_THRESHOLD, df["vader_compound"] <= NEG_THRESHOLD],
        ["Positive", "Negative"],
        default="Neutral",
    )
    df["vader_applicable"] = df["lang"].eq("en")

    dist = df["vader_label"].value_counts(normalize=True)
    log("VADER label distribution (all C_text):")
    for lab in ("Positive", "Neutral", "Negative"):
        log(f"    {lab:9s} {dist.get(lab, 0.0):.1%}")

    en = df[df["vader_applicable"]]
    if len(en):
        dist_en = en["vader_label"].value_counts(normalize=True)
        log("VADER label distribution (English only):")
        for lab in ("Positive", "Neutral", "Negative"):
            log(f"    {lab:9s} {dist_en.get(lab, 0.0):.1%}")
    return df


def main() -> None:
    args = base_parser(__doc__ or "Phase 1A — VADER").parse_args()
    set_seed(args.seed)

    require(MASTER_PATH, "python src/scripts/01_preprocess.py")
    df = pd.read_parquet(MASTER_PATH, columns=["reviewId", "app_name", "content_clean", "lang", "corpus_text"])

    df_text = df[df["corpus_text"]].copy()
    log(f"C_text subset: {len(df_text):,} reviews")
    df_text = maybe_sample(df_text, args)

    df_text = run_vader_sentiment(df_text)

    keep = ["reviewId", "vader_neg", "vader_neu", "vader_pos",
            "vader_compound", "vader_label", "vader_applicable"]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_text[keep].to_parquet(OUTPUT_PATH, index=False, compression="zstd")
    log(f"Saved {len(df_text):,} rows → {OUTPUT_PATH}")

    log("")
    log("Next: python src/scripts/02b_sentiment_roberta.py")


if __name__ == "__main__":
    main()
