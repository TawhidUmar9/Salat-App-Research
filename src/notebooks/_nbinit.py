"""
Shared bootstrap for the analysis notebooks.

Usage at the top of every notebook:

    from _nbinit import *          # noqa: F403

Puts scripts/ on the path, loads the plotting style, and exposes the standard
data loaders so each notebook starts with `reviews`, `aspects`, `features`,
`demand` available without repeating the boilerplate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "scripts"))

from _common import (  # noqa: E402,F401
    ASPECT_PATH, DATA_DIR, DEMAND_PATH, FEATURE_MATRIX_PATH, FIGURES_DIR,
    GOLD_DIR, QUOTES_DIR, SENTIMENT_PATH, TEMPORAL_PATH, TOPICS_PATH,
    ASPECT_TO_FEATURE, CSV_LINKED_ASPECTS, REVIEW_ONLY_ASPECTS,
)

pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_colwidth", 90)


def setup_plots():
    """Load the shared paper style and return (plt, WONG palette)."""
    import matplotlib.pyplot as plt
    style = FIGURES_DIR / "style.mplstyle"
    if style.exists():
        plt.style.use(str(style))
    wong = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
            "#D55E00", "#56B4E9", "#F0E442", "#000000"]
    return plt, wong


def load_reviews(text_only: bool = True) -> pd.DataFrame:
    df = pd.read_parquet(SENTIMENT_PATH)
    if text_only:
        df = df[df["corpus_text"]].copy()
    df["sent_num"] = df["roberta_label"].map({"Negative": -1.0, "Neutral": 0.0, "Positive": 1.0})
    return df


def load_aspects(exclude_requests: bool = True) -> pd.DataFrame:
    a = pd.read_parquet(ASPECT_PATH)
    if exclude_requests:
        a = a[~a["is_request"].fillna(False)]
    return a


def load_features() -> pd.DataFrame:
    return pd.read_parquet(FEATURE_MATRIX_PATH)


def load_demand() -> pd.DataFrame:
    return pd.read_parquet(DEMAND_PATH) if DEMAND_PATH.exists() else pd.DataFrame()


def load_model_results() -> pd.DataFrame:
    p = DATA_DIR / "model_results.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def export_quotes(df: pd.DataFrame, rq: str, n: int = 15,
                  text_col: str = "content_clean") -> pd.DataFrame:
    """
    Export verbatim illustrative quotes for the paper's qualitative passages.

    userName is dropped, never written: §10 requires reviewer names stay out of
    the paper, and the safest way to honour that is to never put them in the file.
    """
    cols = [c for c in ["reviewId", "app_name", "score", "at", text_col] if c in df.columns]
    out = df[cols].head(n).copy()
    out = out.rename(columns={text_col: "quote"})
    out["rq"] = rq
    QUOTES_DIR.mkdir(parents=True, exist_ok=True)
    path = QUOTES_DIR / f"{rq}.csv"
    out.to_csv(path, index=False)
    print(f"Exported {len(out)} quotes → {path}")
    return out


def sentiment_split(series: pd.Series) -> pd.Series:
    """Normalized Positive/Neutral/Negative shares, always in that order."""
    return series.value_counts(normalize=True).reindex(
        ["Positive", "Neutral", "Negative"]
    ).fillna(0.0)


__all__ = [
    "np", "pd", "Path",
    "setup_plots", "load_reviews", "load_aspects", "load_features",
    "load_demand", "load_model_results", "export_quotes", "sentiment_split",
    "ASPECT_PATH", "DATA_DIR", "DEMAND_PATH", "FEATURE_MATRIX_PATH",
    "FIGURES_DIR", "GOLD_DIR", "QUOTES_DIR", "SENTIMENT_PATH",
    "TEMPORAL_PATH", "TOPICS_PATH",
    "ASPECT_TO_FEATURE", "CSV_LINKED_ASPECTS", "REVIEW_ONLY_ASPECTS",
]
