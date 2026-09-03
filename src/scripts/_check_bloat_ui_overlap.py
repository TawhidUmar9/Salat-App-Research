"""
Is RQ5's bloat x ui_design co-occurrence real, or a measurement artefact?

The confirmatory test reports how often the two complaints appear in the same
REVIEW. That is only evidence for the surfacing account if the two tags come
from DIFFERENT sentences — a user complaining about clutter in one breath and
the interface in another. If instead one sentence carries both tags, the
co-occurrence is partly definitional, because these two patterns both match
"hard to use" and "hard to navigate":

    complexity_bloat : (?:hard|difficult|confusing)\\s+to\\s+(?:navigate|find|use|understand)
    ui_design        : (?:...|hard\\s+to\\s+(?:use|navigate))

This splits the co-occurring reviews into those two cases and recomputes the
lift on the defensible subset — reviews where the two tags came from separate
sentences. Report that number, not the raw one.

Usage:
    python src/scripts/_check_bloat_ui_overlap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ASPECT_PATH, log, require, section  # noqa: E402

BLOAT, UI = "complexity_bloat", "ui_design"


def main() -> None:
    require(ASPECT_PATH, "python src/scripts/03_absa.py")
    asp = pd.read_parquet(ASPECT_PATH)
    section("RQ5 co-occurrence — measurement artefact check")

    # Match build_design_matrix exactly: negative AND non-request. An earlier
    # version omitted the request filter and used a different denominator, which
    # is why its lift disagreed with the one 06_models reports.
    neg = asp[asp["sentiment_label"].astype(str).str.lower().eq("negative")
              & (~asp["is_request"].fillna(False))]
    pair = neg[neg["aspect"].isin([BLOAT, UI])]
    if pair.empty:
        log("no negative bloat/ui rows found", level="WARN")
        return

    by_review = pair.groupby("reviewId")["aspect"].agg(set)
    both = by_review[by_review.map(lambda s: {BLOAT, UI} <= s)].index
    log(f"reviews with BOTH complaints: {len(both):,}")

    # For each such review, did any single sentence carry both tags?
    sub = pair[pair["reviewId"].isin(both)]
    per_sentence = (sub.groupby(["reviewId", "triggering_sentence"])["aspect"]
                    .agg(set).reset_index())
    dbl = per_sentence[per_sentence["aspect"].map(lambda s: {BLOAT, UI} <= s)]
    reviews_with_double = set(dbl["reviewId"])

    same = len(reviews_with_double)
    diff = len(both) - same
    log(f"  same sentence double-tagged (definitional): {same:,} "
        f"({same / max(1, len(both)):.1%})")
    log(f"  different sentences (real co-occurrence):   {diff:,} "
        f"({diff / max(1, len(both)):.1%})")

    if same:
        log("")
        log("examples of the double-tagged sentences:")
        for s in dbl["triggering_sentence"].head(5):
            log(f"    {str(s)[:88]}")

    # Recompute the lift excluding definitional co-occurrence.
    n_reviews = asp["reviewId"].nunique()
    n_bloat = neg[neg["aspect"].eq(BLOAT)]["reviewId"].nunique()
    n_ui = neg[neg["aspect"].eq(UI)]["reviewId"].nunique()
    base = n_ui / n_reviews
    log("")
    log(f"P(ui) base rate: {base:.4%}")
    log(f"  lift as reported          : {(len(both) / max(1, n_bloat)) / base:.2f}x")
    log(f"  lift, definitional removed: "
        f"{(diff / max(1, n_bloat)) / base:.2f}x   <- report this one")


if __name__ == "__main__":
    main()
