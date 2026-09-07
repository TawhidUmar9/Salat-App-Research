"""
Draw a cross-app extension to the tracker coding sheet.
=======================================================
The original fifty coded reviews all came from one app: `06_models.py` selects
with `nsmallest(50, "sent_num")`, sent_num is three-valued, so every negative
review ties and pandas returns the first fifty in frame order — app order. That
sheet is a single-app case study and cannot carry a cross-app claim.

This draws a stratified extension from the OTHER tracker apps so the RQ1 themes
can be checked against the ecosystem. It never touches the original sheet.

    - same tracker definition as M2 (prayer_tracker / tracker_score / goal_system)
    - negative model sentiment, same as the original draw
    - reviewIds already coded are excluded
    - equal allocation per app, so no single app can dominate again
    - seeded, so the draw is reproducible

Usage:
    python src/scripts/_draw_tracker_extension.py
    python src/scripts/_draw_tracker_extension.py --n 40 --seed 42
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    ASPECT_PATH, DATA_DIR, SENTIMENT_PATH, log, require, section,
    write_coding_sheet,
)

TRACKER_ASPECTS = ["prayer_tracker", "tracker_score", "goal_system"]
ORIGINAL = DATA_DIR / "quotes" / "rq1_tracker_negative_50.csv"
OUT = DATA_DIR / "quotes" / "rq1_tracker_negative_extension.csv"

#: The frame the original fifty were coded into. Printed so the second pass
#: applies an existing frame rather than deriving a new one — that is what makes
#: this cheap. Add a code only when nothing here fits; a theme growing is a
#: finding, a theme splitting is a problem.
FRAME = [
    ("A", "Redesign as regression"),
    ("B", "Retroactive logging denied"),
    ("C", "The record can't be trusted"),
    ("D", "Normative mismatch"),
    ("E", "Prompting failures break the loop"),
    ("F", "Monetization against religious purpose"),
    ("—", "Excluded (non-complaint)"),
]


def main() -> None:
    args = sys.argv[1:]
    n_total = int(args[args.index("--n") + 1]) if "--n" in args else 30
    seed = int(args[args.index("--seed") + 1]) if "--seed" in args else 42

    section("Cross-app extension to the tracker coding sheet")
    require(SENTIMENT_PATH, "src/scripts/02b_sentiment_roberta.py")
    require(ASPECT_PATH, "src/scripts/03_absa.py")

    df = pd.read_parquet(SENTIMENT_PATH)
    missing = {"reviewId", "app_name", "score", "at", "content_clean",
               "roberta_label"} - set(df.columns)
    if missing:
        raise SystemExit(f"{SENTIMENT_PATH} is missing columns: {sorted(missing)}")
    if "corpus_text" in df.columns:
        df = df[df["corpus_text"]].copy()

    asp = pd.read_parquet(ASPECT_PATH, columns=["reviewId", "aspect"])
    tracker_ids = set(asp.loc[asp["aspect"].isin(TRACKER_ASPECTS), "reviewId"])
    d = df[df["reviewId"].isin(tracker_ids) & (df["roberta_label"] == "Negative")].copy()
    log(f"Tracker reviews with negative sentiment: {len(d):,}")

    already: set = set()
    covered: set = set()
    if ORIGINAL.exists():
        prev = pd.read_csv(ORIGINAL)
        already = set(prev["reviewId"].astype(str))
        covered = set(prev["app_name"])
        log(f"Excluding {len(already)} already-coded reviews from "
            f"{len(covered)} app(s): {', '.join(sorted(covered))}")

    d["reviewId"] = d["reviewId"].astype(str)
    pool = d[~d["reviewId"].isin(already) & ~d["app_name"].isin(covered)]
    apps = sorted(pool["app_name"].unique())
    if not len(apps):
        raise SystemExit("No other tracker apps have negative tracker reviews.")
    log(f"Eligible pool: {len(pool):,} reviews across {len(apps)} other apps")

    # Guarantee a floor per app so no app can dominate the way the original draw
    # did, then top up to n_total from whatever is left. The top-up is drawn from
    # the pooled remainder, so it leans toward apps with more eligible reviews —
    # a supplement to guaranteed coverage, not a replacement for it.
    per = max(1, n_total // len(apps))
    parts = []
    for app in apps:
        sub = pool[pool["app_name"] == app]
        parts.append(sub.sample(n=min(per, len(sub)), random_state=seed))

    chosen = pd.concat(parts)
    if len(chosen) < n_total:
        rest = pool[~pool["reviewId"].isin(set(chosen["reviewId"]))]
        if len(rest):
            parts.append(rest.sample(n=min(n_total - len(chosen), len(rest)),
                                     random_state=seed))

    out = (pd.concat(parts)
             .sort_values(["app_name", "reviewId"])
             .head(n_total)[["reviewId", "app_name", "score", "at", "content_clean"]])

    log("")
    log(f"Drawn {len(out)} reviews across {out['app_name'].nunique()} apps:")
    for app, k in out["app_name"].value_counts().items():
        log(f"    {k:>3d}  {app}")

    out = out.copy()
    out["theme_code"] = ""   # open code, in your own words
    out["theme"] = ""        # letter from the frame below
    write_coding_sheet(out, OUT)

    log("")
    log("Apply the EXISTING frame — do not derive a new one:")
    for k, name in FRAME:
        log(f"    {k}. {name}")
    log("")
    log("Fill `theme_code` with a short phrase and `theme` with the letter above.")
    log("Add a code only when nothing in the frame fits.")


if __name__ == "__main__":
    main()
