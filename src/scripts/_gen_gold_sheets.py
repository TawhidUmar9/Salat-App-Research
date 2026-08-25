"""
Generate every hand-labelling sheet — no Jupyter required.

Mirrors §3 of notebooks/02c_cross_validation.py exactly: same stratification,
same seeds, same columns, so the sheets are identical to the notebook's. This
exists because the notebook calls Jupyter's `display()` and therefore cannot be
run as a plain script, which is awkward on a headless server — and generating
the sheets is the only thing standing between a finished pipeline run and days
of annotation work.

The notebook is still needed LATER, for two things this script does not do:
  * section 4  — the κ / precision / recall / F1 / Krippendorff's α evaluation
  * "Zero-shot threshold calibration" — reads aspect_300.csv back to pick the
    --zeroshot-threshold for the second 03_absa pass

Supersedes _gen_aspect_gold_standalone.py, which generated only aspect_300.

Prerequisites (run 01 → 03 first):
    src/data/master_reviews_with_sentiment.parquet
    src/data/aspect_sentiments.parquet

Usage:
    python src/scripts/_gen_gold_sheets.py
    python src/scripts/_gen_gold_sheets.py --doc-n 500 --aspect-n 300 --seed 42
    python src/scripts/_gen_gold_sheets.py --annotators 1   # no overlap split
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    ASPECT_PATH, GOLD_DIR, SENTIMENT_PATH, log, require, section,
    stratified_sample, write_gold_sheet,
)

#: Rows shared by both annotator files. This overlap is the ONLY source of
#: Krippendorff's α — both people label these same rows independently.
OVERLAP = 100

DOC_ACTION_COLS = ["gold_label", "annotator", "notes"]
ASPECT_ACTION_COLS = ["gold_aspect_correct", "gold_aspect_true", "gold_sentiment",
                      "gold_is_request", "annotator"]


def install_tier(n) -> str:
    if pd.isna(n):
        return "unknown"
    n = float(n)
    if n > 100_000_000:
        return ">100M"
    if n > 10_000_000:
        return "10M-100M"
    if n > 1_000_000:
        return "1M-10M"
    return "<1M"


def star_band(s) -> str:
    if pd.isna(s):
        return "unknown"
    s = float(s)
    return "1-2" if s <= 2 else "3" if s == 3 else "4-5"


def build_doc_sheet(data: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """500 reviews stratified by install tier × star band × language."""
    d = data.copy()
    d["install_tier"] = d["realInstalls"].map(install_tier)
    d["star_band"] = d["score"].map(star_band)
    d["lang_band"] = np.where(d["lang"] == "en", "en", "translated")
    d["stratum"] = d["install_tier"] + "|" + d["star_band"] + "|" + d["lang_band"]

    sample = stratified_sample(d, "stratum", n, seed=seed)
    sheet = sample[["reviewId", "app_name", "score", "lang", "install_tier",
                    "star_band", "content_clean"]].copy()
    sheet["gold_label"] = ""    # ACTION: Positive / Negative / Neutral / Mixed
    sheet["annotator"] = ""
    sheet["notes"] = ""
    return sheet.reset_index(drop=True)


def build_aspect_sheet(n: int, seed: int) -> pd.DataFrame | None:
    """300 sentences stratified BY ASPECT so rare aspects are actually covered."""
    if not ASPECT_PATH.exists():
        log("aspect_sentiments.parquet not found — run 03_absa.py first.", level="WARN")
        return None
    asp = pd.read_parquet(ASPECT_PATH)
    scored = asp[asp["triggering_sentence"].notna()]

    sheet = stratified_sample(scored, "aspect", n, seed=seed)
    sheet = sheet[[
        "reviewId", "app_name", "aspect", "triggering_sentence",
        "sentiment_label", "match_method", "is_request", "score",
    ]].rename(columns={
        "sentiment_label": "predicted_sentiment", "aspect": "predicted_aspect",
    })
    sheet["gold_aspect_correct"] = ""   # ACTION: 1 / 0
    sheet["gold_aspect_true"] = ""      # ACTION: correct aspect name if 0
    sheet["gold_sentiment"] = ""        # ACTION: Positive/Negative/Neutral/Mixed
    sheet["gold_is_request"] = ""       # ACTION: 1 / 0
    sheet["annotator"] = ""
    return sheet.reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--doc-n", type=int, default=500)
    p.add_argument("--aspect-n", type=int, default=300)
    p.add_argument("--annotators", type=int, default=2, choices=[1, 2])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    section("Document-level gold set")
    require(SENTIMENT_PATH, "python src/scripts/02b_sentiment_roberta.py")
    df = pd.read_parquet(SENTIMENT_PATH)
    df = df[df["corpus_text"]].copy()
    log(f"C_text: {len(df):,} reviews across {df['app_name'].nunique()} apps")

    doc = build_doc_sheet(df, args.doc_n, args.seed)
    write_gold_sheet(doc, GOLD_DIR / "doc_500.csv", DOC_ACTION_COLS)
    log(f"Wrote {len(doc)} rows → doc_500.csv  (reference copy — do not label)")

    if args.annotators == 2:
        overlap, rest = doc.head(OVERLAP), doc.iloc[OVERLAP:]
        half = len(rest) // 2
        for i, part in enumerate([rest.iloc[:half], rest.iloc[half:]], start=1):
            out = pd.concat([overlap.assign(is_overlap=True),
                             part.assign(is_overlap=False)], ignore_index=True)
            write_gold_sheet(out, GOLD_DIR / f"doc_500_annotator{i}.csv", DOC_ACTION_COLS)
            log(f"Wrote {len(out)} rows → doc_500_annotator{i}.csv "
                f"({OVERLAP} shared with the other annotator)")
        log("The shared rows are the ONLY source of Krippendorff's α — both "
            "annotators must label them independently, without discussion.")

    section("Aspect-level gold set")
    asp = build_aspect_sheet(args.aspect_n, args.seed)
    if asp is not None:
        write_gold_sheet(asp, GOLD_DIR / "aspect_300.csv", ASPECT_ACTION_COLS)
        log(f"Wrote {len(asp)} rows → aspect_300.csv "
            f"({asp['predicted_aspect'].nunique()} aspects covered)")

    section("Next")
    log(f"Sheets are in {GOLD_DIR}")
    log("  doc_500_annotator1.csv  — you        (gold_label)")
    log("  doc_500_annotator2.csv  — 2nd coder  (gold_label)")
    log("  aspect_300.csv          — you        (4 gold_* columns)")
    log("  demand_precision_sample.csv — you    (is_true_positive), written by 03c")
    log("")
    log("Copy them to a machine with a spreadsheet app, label, copy back, then")
    log("run the notebook's section 4 + threshold calibration.")


if __name__ == "__main__":
    main()
