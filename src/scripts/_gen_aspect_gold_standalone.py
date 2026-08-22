"""
Standalone aspect_300.csv generator — no Jupyter required.

Mirrors 02c_cross_validation.py §3 "Aspect-level gold set" exactly (same
stratified-by-aspect sampling, same columns). Use the real notebook instead
once jupyter/jupytext are installed — it also has the Section 4 evaluation
(precision/recall, threshold calibration) this script doesn't.

Prerequisite: src/data/aspect_sentiments.parquet must exist (run 03_absa.py first).

Usage:
    .venv/bin/python src/scripts/_gen_aspect_gold_standalone.py
    .venv/bin/python src/scripts/_gen_aspect_gold_standalone.py --n 300 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    ASPECT_PATH, GOLD_DIR, log, require, stratified_sample, write_gold_sheet,
)

import pandas as pd  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    require(ASPECT_PATH, "python src/scripts/03_absa.py")
    asp = pd.read_parquet(ASPECT_PATH)
    scored = asp[asp["triggering_sentence"].notna()]

    # Stratify by aspect so rare aspects (women_period, qasr_travel) are
    # covered — a proportional sample would contain almost none of them.
    aspect_sheet = stratified_sample(scored, "aspect", args.n, seed=args.seed)
    aspect_sheet = aspect_sheet[[
        "reviewId", "app_name", "aspect", "triggering_sentence",
        "sentiment_label", "match_method", "is_request", "score",
    ]].rename(columns={
        "sentiment_label": "predicted_sentiment", "aspect": "predicted_aspect",
    })
    aspect_sheet["gold_aspect_correct"] = ""    # ACTION: 1 / 0
    aspect_sheet["gold_aspect_true"] = ""       # ACTION: correct aspect if 0
    aspect_sheet["gold_sentiment"] = ""         # ACTION: Positive/Negative/Neutral/Mixed
    aspect_sheet["gold_is_request"] = ""        # ACTION: 1 / 0
    aspect_sheet["annotator"] = ""

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = write_gold_sheet(
        aspect_sheet, GOLD_DIR / "aspect_300.csv",
        ["gold_aspect_correct", "gold_aspect_true", "gold_sentiment",
         "gold_is_request", "annotator"],
    )
    log(f"Wrote {out_path}  ({len(aspect_sheet)} rows, "
        f"{aspect_sheet['predicted_aspect'].nunique()} aspects)")
    log("Next: fill gold_aspect_correct / gold_aspect_true / gold_sentiment / "
        "gold_is_request by hand, then evaluate (§4 of 02c_cross_validation.py, "
        "or ask for a standalone evaluator).")


if __name__ == "__main__":
    main()
