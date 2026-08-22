"""
Phase 2.5b — Demand Mining
===========================
Extract explicit feature requests, unmet needs, churn signals,
and preference citations from review text.

Input:
    - data/master_reviews_with_sentiment.parquet
    - data/aspect_sentiments.parquet (is_request flags from ABSA)

Output:
    - data/demand.parquet
      Schema: (reviewId, app, aspect, request_type, evidence_sentence)

See implementation_plan.md §6.2 for full specification.

This module is what rescues the rare-feature RQs (§2.1). "Auto Qasr exists in
1 of 20 apps, and N reviews across the other 19 explicitly ask for travel/qasr
handling" is a stronger and more publishable finding than a null between-app
rating test.

Usage:
    python src/scripts/03c_demand_mining.py
    python src/scripts/03c_demand_mining.py --sample 20000
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    ASPECT_PATH, DEMAND_PATH, GOLD_DIR, SENTIMENT_PATH,
    base_parser, compile_aspect_patterns, load_lexicon, log, maybe_sample,
    require, section, set_seed, stratified_sample, summarize, write_gold_sheet,
)

# ─── Pattern categories (§6.2) ──────────────────────────────────────────────────

#: Explicit feature requests.
REQUEST_PATTERNS = [
    r"\bwish(?:es|ed)?\b",
    r"\bhope(?:s|d)?\s+(?:you|they|to\s+see|for)\b",
    r"\bplease\s+(?:add|include|provide|make|give|implement)\b",
    r"\b(?:pls|plz)\s+(?:add|include|make|give)\b",
    r"\bkindly\s+(?:add|include|provide|make)\b",
    r"\bshould\s+(?:have|add|include|support|be\s+able)\b",
    r"\bwould\s+be\s+(?:nice|better|great|good|helpful|perfect)\s+if\b",
    r"\bit\s+would\s+be\s+(?:nice|better|great|good|helpful)\b",
    r"\b(?:can|could|would)\s+(?:you|u)\s+(?:please\s+)?(?:add|include|make|provide)\b",
    r"\bneeds?\s+(?:a|an|to\s+have|more|better)\b",
    r"\bmissing\b",
    r"\bno\s+option\s+(?:for|to)\b",
    r"\bsuggest(?:ion|ions)?\b",
    r"\black(?:s|ing)?\b",
    r"\brequest(?:ing)?\b",
    r"\bwant(?:ed)?\s+(?:a|an|to\s+see)\b",
    r"\bhopefully\s+(?:you|they|add)\b",
]

#: Stating a feature does not exist.
ABSENCE_PATTERNS = [
    r"\bdoes\s?n[o']?t\s+(?:have|support|include|offer|work\s+with)\b",
    r"\bdo\s+not\s+(?:have|support|include|offer)\b",
    r"\bdid\s?n[o']?t\s+(?:have|find|see)\b",
    r"\bthere\s+is\s+no\b",
    r"\bthere'?s\s+no\b",
    r"\bhas\s+no\b",
    r"\bwith\s?out\s+(?:a|an|any)\b",
    r"\bcould\s?n[o']?t\s+find\b",
    r"\bcan\s?n[o']?t\s+find\b",
    r"\bunable\s+to\s+find\b",
    r"\bnot\s+available\b",
    r"\bnot\s+supported\b",
]

#: Switching / churn — feeds RQ4 loyalty and RQ3 competitive edge.
CHURN_PATTERNS = [
    r"\buninstall(?:ed|ing)?\b",
    r"\bdelet(?:e|ed|ing)\s+(?:this|the\s+app|it)\b",
    r"\bswitch(?:ed|ing)?\s+to\b",
    r"\bmov(?:ed|ing)\s+to\b",
    r"\bgoing\s+back\s+to\b",
    r"\bwent\s+back\s+to\b",
    r"\bbetter\s+than\s+(?:this|the\s+other)\b",
    r"\bfound\s+(?:a\s+)?better\b",
    r"\bremov(?:ed|ing)\s+(?:this|the\s+app|it)\b",
    r"\blooking\s+for\s+(?:another|an\s+alternative)\b",
]

#: Preference citations — the operationalization of "competitive edge" (RQ3).
PREFERENCE_PATTERNS = [
    r"\bthat'?s\s+why\s+i\s+(?:use|keep|chose|prefer)\b",
    r"\bonly\s+app\s+(?:that|which|with)\b",
    r"\bno\s+other\s+app\b",
    r"\bthe\s+(?:only\s+)?reason\s+i\s+(?:use|chose|keep|downloaded)\b",
    r"\bbest\s+app\s+(?:for|because|that)\b",
    r"\bbetter\s+than\s+(?:all\s+)?(?:other|the\s+rest)\b",
    r"\bi\s+(?:use|chose|prefer)\s+this\s+(?:app\s+)?because\b",
    r"\bunlike\s+other\s+apps\b",
    r"\bwhat\s+(?:sets|makes)\s+(?:this|it)\s+apart\b",
]

#: Loyalty tenure proxies (§9.3). Extracted here so M3 can consume them.
TENURE_PATTERNS = [
    r"\busing\s+(?:this|it|the\s+app)\s+(?:for\s+)?(?:the\s+(?:past|last)\s+)?\d+\s*(?:\+\s*)?(?:year|yr|month|mo)s?\b",
    r"\bsince\s+20\d\d\b",
    r"\bfor\s+(?:the\s+)?(?:past|last)\s+\d+\s*(?:year|yr|month|mo)s?\b",
    r"\b\d+\s*(?:\+\s*)?years?\s+(?:of\s+)?(?:use|using|now)\b",
]

PATTERN_SETS: dict[str, list[str]] = {
    "request": REQUEST_PATTERNS,
    "absence": ABSENCE_PATTERNS,
    "churn": CHURN_PATTERNS,
    "preference": PREFERENCE_PATTERNS,
    "tenure": TENURE_PATTERNS,
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।؟])\s+|\n+|(?<=,)\s+(?=(?:but|and|however|though)\b)")


def _compile(patterns: list[str]) -> re.Pattern:
    return re.compile("|".join(patterns), re.IGNORECASE | re.UNICODE)


COMPILED = {name: _compile(pats) for name, pats in PATTERN_SETS.items()}


def explode_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split reviews into sentences for evidence attribution.

    Demand matching happens at sentence level so `evidence_sentence` is a
    quotable unit rather than a whole review, and so an aspect keyword in one
    clause is not paired with a request pattern three sentences away.
    """
    section("Sentence explosion")

    rid, app, sent, star = [], [], [], []
    for r, a, t, s in zip(
        df["reviewId"], df["app_name"], df["content_clean"].fillna(""), df["score"]
    ):
        if not t.strip():
            continue
        for part in _SENTENCE_SPLIT.split(t):
            part = part.strip()
            if len(part.split()) >= 3:
                rid.append(r)
                app.append(a)
                sent.append(part)
                star.append(s)

    out = pd.DataFrame({
        "reviewId": rid, "app_name": app, "evidence_sentence": sent, "score": star,
    })
    out["_lower"] = out["evidence_sentence"].str.lower()
    log(f"{len(df):,} reviews → {len(out):,} candidate sentences")
    return out


def _extract(sentences: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Apply one pattern set and return the matching sentences."""
    pat = COMPILED[kind]
    mask = sentences["_lower"].str.contains(pat, regex=True, na=False)
    out = sentences.loc[mask, ["reviewId", "app_name", "evidence_sentence", "score"]].copy()
    out["request_type"] = kind
    log(f"    {kind:12s} {len(out):>8,} sentences "
        f"({out['reviewId'].nunique():,} reviews)")
    return out


def extract_request_patterns(sentences: pd.DataFrame) -> pd.DataFrame:
    """§6.2 — Explicit feature requests."""
    return _extract(sentences, "request")


def extract_absence_patterns(sentences: pd.DataFrame) -> pd.DataFrame:
    """§6.2 — Statements that a feature does not exist."""
    return _extract(sentences, "absence")


def extract_churn_signals(sentences: pd.DataFrame) -> pd.DataFrame:
    """§6.2 — Switching and uninstall signals; feeds RQ4 loyalty and RQ3."""
    return _extract(sentences, "churn")


def extract_preference_citations(sentences: pd.DataFrame) -> pd.DataFrame:
    """§6.2 — Preference citations; the competitive-edge measure for RQ3."""
    return _extract(sentences, "preference")


def extract_tenure_mentions(sentences: pd.DataFrame) -> pd.DataFrame:
    """§9.3 — Tenure mentions, a labelled *proxy* for loyalty."""
    return _extract(sentences, "tenure")


def attach_aspects(demand: pd.DataFrame, lexicon: dict) -> pd.DataFrame:
    """
    Attach each demand match to the aspect(s) its sentence mentions.

    A sentence with no aspect keyword is kept with aspect = NA: "please add
    dark mode" is still demand evidence even when it falls outside the 27-aspect
    taxonomy, and the unattached volume is worth reporting.
    """
    section("Attaching aspects")

    patterns = compile_aspect_patterns(lexicon)
    low = demand["evidence_sentence"].str.lower()

    frames = []
    for aspect, pat in patterns.items():
        mask = low.str.contains(pat, regex=True, na=False)
        if mask.any():
            sub = demand.loc[mask].copy()
            sub["aspect"] = aspect
            frames.append(sub)

    matched = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    key = ["reviewId", "evidence_sentence", "request_type"]
    if len(matched):
        seen = set(map(tuple, matched[key].itertuples(index=False, name=None)))
        unmatched_mask = [tuple(r) not in seen for r in demand[key].itertuples(index=False, name=None)]
    else:
        unmatched_mask = [True] * len(demand)

    unmatched = demand.loc[unmatched_mask].copy()
    unmatched["aspect"] = pd.NA

    out = pd.concat([matched, unmatched], ignore_index=True)
    log(f"Aspect-attached: {len(matched):,} rows; unattached: {len(unmatched):,} rows")

    if len(matched):
        log("Top aspects by demand volume:")
        for aspect, n in matched["aspect"].value_counts().head(15).items():
            log(f"    {aspect:24s} {n:>7,}")
    return out


def precision_check_sample(demand_df: pd.DataFrame, sample_size: int = 200) -> pd.DataFrame:
    """
    §6.2 — Generate a manual precision-check sheet.

    These regex patterns are noisy and the paper must report their precision.
    The sheet is stratified across request_type so every pattern family gets
    checked, not just the most frequent one.
    """
    section("§6.2  Precision-check sample")

    sample = stratified_sample(demand_df, "request_type", sample_size, seed=0)

    sheet = sample[["reviewId", "app_name", "request_type", "aspect", "evidence_sentence", "score"]].copy()
    sheet["is_true_positive"] = ""      # annotator fills: 1 / 0
    sheet["correct_aspect"] = ""        # annotator fills if the aspect is wrong
    sheet["notes"] = ""

    out_path = write_gold_sheet(
        sheet, GOLD_DIR / "demand_precision_sample.csv",
        ["is_true_positive", "correct_aspect", "notes"],
    )
    log(f"Wrote {len(sheet)} rows → {out_path}")
    log("ACTION: fill is_true_positive (1/0) and report precision per request_type in the paper.")
    return sheet


def summarize_demand(demand: pd.DataFrame) -> None:
    """Log the headline demand numbers that RQ3/RQ4/RQ6b will report."""
    section("Demand summary")

    log(f"Total demand signals: {len(demand):,} across {demand['reviewId'].nunique():,} reviews")
    log("")
    log("By request type:")
    for kind, n in demand["request_type"].value_counts().items():
        log(f"    {kind:12s} {n:>8,}")

    rare = ["qasr_travel", "women_period", "companion_hardware", "mosque_finder", "calendar_sync"]
    log("")
    log("Rare-feature demand (the §2.1 rescue — these drive RQ3/RQ4/RQ5):")
    for aspect in rare:
        sub = demand[demand["aspect"] == aspect]
        if len(sub):
            log(f"    {aspect:20s} {len(sub):>6,} signals across "
                f"{sub['app_name'].nunique():>2} apps, {sub['reviewId'].nunique():,} reviews")
        else:
            log(f"    {aspect:20s} {0:>6,} signals")


def main() -> None:
    p = base_parser(__doc__ or "Phase 2.5b — demand mining")
    p.add_argument("--precision-sample", type=int, default=200,
                   help="Rows to write to the manual precision-check sheet.")
    args = p.parse_args()
    set_seed(args.seed)

    require(SENTIMENT_PATH, "python src/scripts/02b_sentiment_roberta.py")
    df = pd.read_parquet(
        SENTIMENT_PATH,
        columns=["reviewId", "app_name", "content_clean", "score", "lang", "corpus_text"],
    )
    df = df[df["corpus_text"]].copy()
    log(f"C_text subset: {len(df):,} reviews")
    df = maybe_sample(df, args)

    lexicon = load_lexicon()
    sentences = explode_sentences(df)

    section("§6.2  Pattern extraction")
    parts = [
        extract_request_patterns(sentences),
        extract_absence_patterns(sentences),
        extract_churn_signals(sentences),
        extract_preference_citations(sentences),
        extract_tenure_mentions(sentences),
    ]
    demand = pd.concat(parts, ignore_index=True)
    demand = attach_aspects(demand, lexicon)

    # Cross-check against the ABSA is_request flag: the two are independent
    # implementations of "this is a request", so their overlap is a sanity signal.
    if ASPECT_PATH.exists():
        asp = pd.read_parquet(ASPECT_PATH, columns=["reviewId", "aspect", "is_request"])
        flagged = set(map(tuple, asp[asp["is_request"]][["reviewId", "aspect"]].itertuples(index=False, name=None)))
        pairs = list(demand[["reviewId", "aspect"]].itertuples(index=False, name=None))
        demand["absa_flagged_request"] = [p in flagged for p in pairs]
        req = demand[demand["request_type"] == "request"]
        if len(req):
            log(f"Agreement with ABSA is_request on request-type rows: "
                f"{req['absa_flagged_request'].mean():.1%}")
    else:
        log("aspect_sentiments.parquet not found — skipping ABSA cross-check.", level="WARN")
        demand["absa_flagged_request"] = False

    summarize_demand(demand)
    precision_check_sample(demand, args.precision_sample)

    section("Writing output")
    demand = demand.drop(columns=[c for c in ("_lower",) if c in demand.columns])
    demand.to_parquet(DEMAND_PATH, index=False, compression="zstd")
    summarize(demand, "demand")
    log(f"Saved → {DEMAND_PATH}")
    log("")
    log("Next: python src/scripts/04_topic_modeling.py")


if __name__ == "__main__":
    main()
