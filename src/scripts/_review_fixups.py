"""Dump every number still outstanding after the CHI review pass.

Run on the machine that has the Parquet files:

    python src/scripts/_review_fixups.py > review_fixups.txt 2>&1

Then send review_fixups.txt back. Each block is independent and failures are
caught, so a missing input degrades that block only.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    ASPECT_PATH, DATA_DIR, SENTIMENT_PATH, LEXICON_PATH,
)


def block(name):
    def deco(fn):
        print("\n" + "=" * 72)
        print(name)
        print("=" * 72)
        try:
            fn()
        except Exception:
            print("FAILED:")
            traceback.print_exc(limit=3)
        return fn
    return deco


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


@block("1. ASPECT PRECISION TABLE  (all aspects, gold n, Wilson CI, corpus n)")
def _precision():
    import _aspect_precision as ap  # noqa
    from _common import compile_aspect_patterns, load_lexicon
    import importlib
    absa = importlib.import_module("03_absa")

    gold = pd.read_csv(Path(DATA_DIR) / "gold_labels" / "aspect_300.csv")
    gold = gold.rename(columns={"predicted_aspect": "aspect"})
    gold["ok"] = gold["gold_aspect_correct"].astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "y"}
    )
    pats = compile_aspect_patterns(load_lexicon())

    def survives(row):
        s = str(row["triggering_sentence"])
        if str(row.get("match_method", "")).strip() == "zero-shot":
            return False
        if absa.is_devotional(s):
            return False
        pat = pats.get(row["aspect"])
        return bool(pat.search(s)) if pat is not None else True

    g = gold[gold.apply(survives, axis=1)].copy()
    freq = pd.read_parquet(ASPECT_PATH, columns=["aspect"])["aspect"].value_counts()

    print(f"gold rows labelled {len(gold)} -> surviving {len(g)}")
    print(f"unweighted precision over survivors: {g['ok'].mean():.1%}")
    print(f"{'aspect':<24}{'corpus_n':>10}{'gold_n':>8}{'prec':>7}   95% CI")
    rows = g.groupby("aspect")["ok"].agg(["sum", "count", "mean"])
    rows["corpus_n"] = rows.index.map(freq).fillna(0).astype(int)
    for a, r in rows.sort_values("corpus_n", ascending=False).iterrows():
        lo, hi = wilson(int(r["sum"]), int(r["count"]))
        adj = r["mean"] * r["corpus_n"]
        print(f"{a:<24}{int(r['corpus_n']):>10,}{int(r['count']):>8}"
              f"{r['mean']:>7.0%}   [{lo:.0%}, {hi:.0%}]"
              f"   adjusted={adj:,.0f}  interval=[{lo*r['corpus_n']:,.0f}, {hi*r['corpus_n']:,.0f}]")
    miss = sorted(set(freq.index) - set(rows.index))
    print(f"\naspects with corpus volume but NO surviving gold rows: {miss}")


@block("2. DEVOTIONAL FILTER  (denominator, unit, pipeline position)")
def _devotional():
    asp = pd.read_parquet(ASPECT_PATH)
    print(f"aspect table rows (aspect, sentence pairs): {len(asp):,}")
    print(f"distinct reviewIds in aspect table:         {asp['reviewId'].nunique():,}")
    if "triggering_sentence" in asp.columns:
        print(f"distinct triggering sentences:              "
              f"{asp['triggering_sentence'].nunique():,}")
    import importlib
    absa = importlib.import_module("03_absa")
    df = pd.read_parquet(SENTIMENT_PATH, columns=["reviewId", "content"])
    sents = absa.segment_sentences(df)
    print(f"sentence corpus (segment_sentences output):  {len(sents):,}")
    dev = sents["sentence"].map(absa.is_devotional).sum()
    print(f"devotional sentences in that corpus:         {dev:,}")
    print("NOTE: paper currently says '1,686 sentences (0.8% of the sentence-level")
    print("corpus)'. Report which of the above is the 1,686 denominator.")


@block("3. RQ4 women_period UNITS  (2,194 vs 2,182 vs gap-matrix basis)")
def _rq4():
    asp = pd.read_parquet(ASPECT_PATH)
    w = asp[asp["aspect"] == "women_period"]
    print(f"aspect-tagged rows (sentences):  {len(w):,}")
    print(f"distinct reviews:                {w['reviewId'].nunique():,}")
    print(f"apps with any:                   {w['app_name'].nunique()}")
    gm = pd.read_csv(Path(DATA_DIR) / "gap_matrix.csv")
    gw = gm[gm["aspect"] == "women_period"]
    print(f"gap_matrix n_mentions total:     {gw['n_mentions'].sum():,.0f}")
    print(f"gap_matrix demand_signals total: {gw['demand_signals'].sum():,.0f}")


@block("4. RQ1 TRACKER SET  (review-level composition of the 6,285)")
def _rq1():
    asp = pd.read_parquet(ASPECT_PATH)
    T = ["prayer_tracker", "tracker_score", "goal_system"]
    ids = {a: set(asp.loc[asp["aspect"] == a, "reviewId"]) for a in T}
    union = set().union(*ids.values())
    only_weak = (ids["tracker_score"] | ids["goal_system"]) - ids["prayer_tracker"]
    print(f"union of the three tags (reviews):       {len(union):,}")
    for a in T:
        print(f"  reviews carrying {a:<16}: {len(ids[a]):,}")
    print(f"reviews carrying ONLY the two weak tags:  {len(only_weak):,} "
          f"({len(only_weak)/max(len(union),1):.1%} of the union)")


@block("5. RQ1 GUILT vs MOTIVATION  (2x2 cells for the Fisher test)")
def _guilt():
    import re
    asp = pd.read_parquet(ASPECT_PATH)
    src = Path(__file__).with_name("06_models.py").read_text()
    pats = re.findall(r'(guilt|motiv)\w*\s*=\s*r?["\'](.+?)["\']', src)
    print("regexes found in 06_models.py:")
    for k, v in pats:
        print(f"  {k}: {v[:110]}")
    print("\nReport the 2x2 contingency table used for OR = 0.283 (n = 149):")
    print("  rows = tracker / non-tracker, cols = guilt-coded / motivation-coded")


@block("6. RQ5 CO-OCCURRENCE CELLS  (the 134 / 94 / 40 base)")
def _rq5():
    asp = pd.read_parquet(ASPECT_PATH)
    b = set(asp.loc[asp["aspect"] == "complexity_bloat", "reviewId"])
    u = set(asp.loc[asp["aspect"] == "ui_design", "reviewId"])
    print(f"reviews with complexity_bloat: {len(b):,}")
    print(f"reviews with ui_design:        {len(u):,}")
    print(f"reviews with BOTH:             {len(b & u):,}")
    same = asp[asp["aspect"].isin(["complexity_bloat", "ui_design"])]
    if "triggering_sentence" in same.columns:
        gp = same.groupby(["reviewId", "triggering_sentence"])["aspect"].nunique()
        fused = gp[gp > 1].index.get_level_values(0).nunique()
        print(f"reviews where both tags share a sentence (fused): {fused:,}")
        print(f"reviews where they are in different sentences:    {len(b & u) - fused:,}")


print("\nDone. Send this whole file back.")
