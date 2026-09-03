"""
Aspect precision, unweighted and corpus-weighted, from the 300-row gold set.

The gold sheet is stratified BY ASPECT — roughly 11 rows each — so its raw
precision treats `qasr_travel` (646 sentences in the corpus) as equal in weight
to `reminders_adhan` (33,502). That is the right design for finding which
aspects fail, and the wrong number to quote for "how accurate is the tagger",
because it is dominated by rare aspects the corpus rarely uses.

Both figures belong in the paper:

  unweighted  — how the tagger does across the taxonomy, rare aspects included.
                The honest worst case, and the one that shows where it breaks.
  weighted    — the expected precision of a randomly drawn aspect assignment,
                which is what every downstream count actually rests on.

VALIDITY NOTE. The gold rows were labelled against the earlier configuration
(zero-shot on, broader lexicon). Every change since then only REMOVES matches —
zero-shot disabled, keywords narrowed, du'a filtered — so the current config's
predictions are a subset of the labelled ones, and filtering the gold set to the
surviving rows is an unbiased estimate rather than a re-labelling. The one
exception is `hard to figure`, newly matched by the narrowed bloat regex, which
no gold row covers. Say so in Methods rather than implying the sample was
re-labelled.

Usage:
    python src/scripts/_aspect_precision.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ASPECT_PATH, GOLD_DIR, log, require, section  # noqa: E402

N_BOOT = 5000


def main() -> None:
    gold_path = GOLD_DIR / "aspect_300.csv"
    require(gold_path, "python src/scripts/_gen_gold_sheets.py")
    require(ASPECT_PATH, "python src/scripts/03_absa.py")

    g = pd.read_csv(gold_path, dtype=str, keep_default_na=False)
    g = g[g["gold_aspect_correct"].str.strip().isin(["0", "1"])].copy()
    g["ok"] = g["gold_aspect_correct"].str.strip() == "1"
    g["aspect"] = g["predicted_aspect"].str.strip()

    asp = pd.read_parquet(ASPECT_PATH, columns=["aspect"])
    freq = asp["aspect"].value_counts()

    section("Aspect precision")
    log(f"gold rows: {len(g)}   aspects covered: {g['aspect'].nunique()}")

    per = g.groupby("aspect")["ok"].agg(["sum", "count", "mean"])
    per["corpus_n"] = per.index.map(freq).fillna(0).astype(int)
    per = per.sort_values("corpus_n", ascending=False)

    unweighted = float(g["ok"].mean())

    # Weighted by corpus frequency, over aspects the gold set actually covers.
    cov = per[per["corpus_n"] > 0]
    w = cov["corpus_n"].to_numpy(float)
    weighted = float(np.average(cov["mean"].to_numpy(float), weights=w))

    # Bootstrap the weighted figure by resampling gold rows within each aspect,
    # which is where the uncertainty lives (~11 rows per aspect).
    rng = np.random.default_rng(0)
    groups = {a: d["ok"].to_numpy() for a, d in g.groupby("aspect")}
    aspects = list(cov.index)
    wts = np.array([cov.at[a, "corpus_n"] for a in aspects], dtype=float)
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        means = np.array([
            rng.choice(groups[a], size=len(groups[a]), replace=True).mean()
            for a in aspects
        ])
        boot[i] = np.average(means, weights=wts)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    log("")
    log(f"  UNWEIGHTED precision : {unweighted:.1%}   (n={len(g)}, aspect-balanced)")
    log(f"  CORPUS-WEIGHTED      : {weighted:.1%}   95% CI [{lo:.1%}, {hi:.1%}]")
    log("")
    log("  by aspect, ordered by corpus volume:")
    log(f"    {'aspect':<24}{'corpus n':>10}{'gold':>7}{'precision':>11}")
    for a, r in per.iterrows():
        log(f"    {a:<24}{int(r['corpus_n']):>10,}{int(r['count']):>7}"
            f"{r['mean']:>10.0%}")

    missing = sorted(set(freq.index) - set(per.index))
    if missing:
        log("")
        log(f"  aspects in the corpus with no gold rows: {missing}", level="WARN")
        share = freq[missing].sum() / freq.sum()
        log(f"  they are {share:.1%} of corpus volume — excluded from the weighted "
            f"figure, so state the coverage.", level="WARN")


if __name__ == "__main__":
    main()
