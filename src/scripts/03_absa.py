"""
Phase 2 — Aspect-Based Sentiment Analysis (ABSA)
=================================================
Sentence-level aspect extraction and per-aspect sentiment.

Input:
    - data/master_reviews_with_sentiment.parquet
    - data/aspect_lexicon.yaml (aspect taxonomy with seed keywords)

Output:
    - data/aspect_sentiments.parquet
      Long format: one row per (reviewId, aspect)
      Columns: sentiment_label, prob_neg, prob_neu, prob_pos,
               triggering_sentence, match_method (lexicon/zero-shot),
               is_request (boolean)

See implementation_plan.md §5 for full specification.

Usage:
    python src/scripts/03_absa.py --sample 20000        # dev run
    python src/scripts/03_absa.py                       # lexicon + zero-shot
    python src/scripts/03_absa.py --no-zero-shot        # lexicon only (fast)
    python src/scripts/03_absa.py --zeroshot-topk 0     # all 27 labels, no prefilter
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    ALL_ASPECTS, ASPECT_DESCRIPTIONS, ASPECT_PATH, CSV_LINKED_ASPECTS,
    REVIEW_ONLY_ASPECTS, SENTIMENT_PATH,
    Checkpoint, auto_batch_size, base_parser, compile_aspect_patterns, get_device,
    load_lexicon, log, maybe_sample, require, section, set_seed, summarize,
)
from _nlp import SequenceClassifier, ZeroShotScorer  # noqa: E402

__all__ = ["ALL_ASPECTS", "CSV_LINKED_ASPECTS", "REVIEW_ONLY_ASPECTS", "ASPECT_DESCRIPTIONS"]

# ─── §5.3 step 5 — request and negation patterns ───────────────────────────────

#: A sentence matching one of these is a feature REQUEST, not a complaint about
#: an existing feature. Scoring "wish it had a qibla" as negative qibla sentiment
#: is the paper-killing failure mode §5.3 warns about.
REQUEST_RE = re.compile(
    r"\b("
    r"wish(es|ed)?|hope(s|d)?|please\s+add|pls\s+add|plz\s+add|kindly\s+add"
    r"|should\s+(have|add|include|be)|would\s+be\s+(nice|better|great|good)\s+if"
    r"|it\s+would\s+be\s+(nice|better|great|good)|can\s+(you|u)\s+(please\s+)?add"
    r"|could\s+you\s+add|need(s)?\s+(a|an|to|more)|missing|no\s+option"
    r"|suggestion|suggest|lacks?|lacking|request(ing)?|add\s+(a|an|more|the)"
    r"|want(ed)?\s+(a|an|to\s+see)|looking\s+forward\s+to"
    r")\b",
    re.IGNORECASE,
)

#: Negation scope: the aspect is stated as absent rather than judged.
NEGATION_RE = re.compile(
    r"\b(doesn'?t|does\s+not|don'?t|do\s+not|didn'?t|did\s+not|isn'?t|is\s+not"
    r"|aren'?t|are\s+not|can'?t|cannot|couldn'?t|won'?t|no|not|never|without|lacks?)\b",
    re.IGNORECASE,
)

#: Rows per checkpoint shard for the sentiment stage.
SHARD_SIZE = 200_000

#: Sentences shorter than this cannot carry aspect content; skip them in the
#: zero-shot stage where every sentence is expensive.
MIN_SENTENCE_WORDS_FOR_ZEROSHOT = 3


# ─── Lexicon ────────────────────────────────────────────────────────────────────

def load_aspect_lexicon() -> dict:
    """
    Load the aspect taxonomy from data/aspect_lexicon.yaml.

    The YAML carries transliteration variants (azan/athan/adhan, zikr/dhikr,
    mazhab/madhab, qaza/qada) and Bengali/Arabic surface forms so the
    untranslated path still matches.
    """
    section("§5.1–5.2  Aspect lexicon")
    lex = load_lexicon()

    unknown = set(lex) - set(ALL_ASPECTS)
    missing = set(ALL_ASPECTS) - set(lex)
    if unknown:
        log(f"Lexicon defines aspects outside the taxonomy: {sorted(unknown)}", level="WARN")
    if missing:
        log(f"Taxonomy aspects missing from the lexicon: {sorted(missing)}", level="WARN")
    return lex


# ─── §5.3 step 1 — sentence segmentation ───────────────────────────────────────

_FALLBACK_SPLIT = re.compile(r"(?<=[.!?।؟])\s+|\n+")


def segment_sentences(df: pd.DataFrame, *, fast: bool = False) -> pd.DataFrame:
    """
    §5.3 Step 1 — Sentence segmentation.

    Aspect assignment and sentiment both happen at sentence level; that is
    what makes "5 stars but the qibla is broken" recoverable.

    pysbd handles informal text well but costs ~1 ms/review. `--fast-split`
    swaps in a punctuation regex for dev runs.
    """
    section("§5.3.1  Sentence segmentation")

    texts = df["content_clean"].fillna("").tolist()
    ids = df["reviewId"].tolist()

    if fast:
        log("Using regex splitter (--fast-split).")
        split_fn = lambda t: [s for s in _FALLBACK_SPLIT.split(t) if s and s.strip()]  # noqa: E731
    else:
        try:
            import pysbd
            seg = pysbd.Segmenter(language="en", clean=False)
            split_fn = seg.segment
            log("Using pysbd segmenter.")
        except ImportError:
            log("pysbd not installed — falling back to regex splitter.", level="WARN")
            split_fn = lambda t: [s for s in _FALLBACK_SPLIT.split(t) if s and s.strip()]  # noqa: E731

    from tqdm.auto import tqdm

    rid_out, sidx_out, sent_out = [], [], []
    for rid, text in zip(tqdm(ids, desc="segment", unit="review", file=sys.stderr), texts):
        if not text.strip():
            continue
        try:
            parts = split_fn(text)
        except Exception:  # noqa: BLE001 — pysbd chokes on rare malformed input
            parts = _FALLBACK_SPLIT.split(text)
        for i, s in enumerate(parts):
            s = s.strip()
            if s:
                rid_out.append(rid)
                sidx_out.append(i)
                sent_out.append(s)

    out = pd.DataFrame({
        "reviewId": rid_out,
        "sentence_index": np.array(sidx_out, dtype="int16"),
        "sentence": sent_out,
    })
    out["sentence_lower"] = out["sentence"].str.lower()
    log(f"{len(df):,} reviews → {len(out):,} sentences "
        f"({len(out) / max(1, len(df)):.1f} per review)")
    return out


# ─── §5.3 step 2 — lexicon tagging ─────────────────────────────────────────────

def lexicon_tagging(sentences_df: pd.DataFrame, lexicon: dict) -> pd.DataFrame:
    """
    §5.3 Step 2 — Lexicon tagging.

    One compiled alternation regex per aspect, applied vectorized over the
    lowercased sentence column. A sentence may carry multiple aspects.
    """
    section("§5.3.2  Lexicon tagging")

    patterns = compile_aspect_patterns(lexicon)
    lower = sentences_df["sentence_lower"]

    hits = []
    for aspect, pat in patterns.items():
        mask = lower.str.contains(pat, regex=True, na=False)
        n = int(mask.sum())
        if n:
            sub = sentences_df.loc[mask, ["reviewId", "sentence_index"]].copy()
            sub["aspect"] = aspect
            hits.append(sub)
        log(f"    {aspect:24s} {n:>8,} sentences ({n / len(lower):.2%})")

    if not hits:
        return pd.DataFrame(columns=["reviewId", "sentence_index", "aspect", "match_method"])

    tagged = pd.concat(hits, ignore_index=True)
    tagged["match_method"] = "lexicon"

    covered = tagged[["reviewId", "sentence_index"]].drop_duplicates()
    log(f"Lexicon hits: {len(tagged):,} (aspect, sentence) pairs "
        f"covering {len(covered):,} / {len(sentences_df):,} sentences "
        f"({len(covered) / len(sentences_df):.1%})")
    return tagged


# ─── §5.3 step 3 — zero-shot backstop ──────────────────────────────────────────

def zero_shot_backstop(
    sentences_df: pd.DataFrame, tagged_df: pd.DataFrame, *,
    device: str, batch_size: int, fp32: bool, threshold: float,
    top_k: int, max_sentences: int | None,
) -> pd.DataFrame:
    """
    §5.3 Step 3 — Zero-shot classification for sentences with no lexicon hit.

    Runs facebook/bart-large-mnli only on unmatched sentences, which keeps the
    cost bounded. See _nlp.ZeroShotScorer for the optimizations that make this
    tractable at corpus scale (dedup, embedding prefilter, length bucketing).

    `threshold` should be calibrated on the 300-review aspect gold set
    (§4.4); 0.75 is a reasonable starting point but is not a substitute for
    calibration.
    """
    section("§5.3.3  Zero-shot backstop")

    covered = set(zip(tagged_df["reviewId"], tagged_df["sentence_index"]))
    key = list(zip(sentences_df["reviewId"], sentences_df["sentence_index"]))
    unmatched = sentences_df[[k not in covered for k in key]].copy()

    n_words = unmatched["sentence"].str.split().map(len)
    unmatched = unmatched[n_words >= MIN_SENTENCE_WORDS_FOR_ZEROSHOT]
    log(f"Unmatched sentences eligible for zero-shot: {len(unmatched):,}")

    if max_sentences and len(unmatched) > max_sentences:
        unmatched = unmatched.sample(max_sentences, random_state=0)
        log(f"Capped to {max_sentences:,} (--zeroshot-max).")

    if unmatched.empty:
        return pd.DataFrame(columns=["reviewId", "sentence_index", "aspect", "match_method"])

    scorer = ZeroShotScorer(
        labels=ASPECT_DESCRIPTIONS, device=device, batch_size=batch_size,
        top_k=top_k, fp32=fp32,
    ).load()

    scores = scorer.score(unmatched["sentence"].tolist())
    names = np.array(scorer.label_names)

    rid, sidx, asp, conf = [], [], [], []
    rid_arr = unmatched["reviewId"].to_numpy()
    sidx_arr = unmatched["sentence_index"].to_numpy()
    for i, row in enumerate(scores):
        for j in np.where(row >= threshold)[0]:
            rid.append(rid_arr[i])
            sidx.append(sidx_arr[i])
            asp.append(names[j])
            conf.append(float(row[j]))

    out = pd.DataFrame({
        "reviewId": rid, "sentence_index": np.array(sidx, dtype="int16"),
        "aspect": asp, "match_method": "zero-shot", "zeroshot_score": conf,
    })
    log(f"Zero-shot added {len(out):,} (aspect, sentence) pairs "
        f"at threshold {threshold} across {out['reviewId'].nunique():,} reviews.")
    if len(out):
        log("Zero-shot aspect distribution:")
        for aspect, n in out["aspect"].value_counts().head(10).items():
            log(f"    {aspect:24s} {n:>8,}")
    return out


# ─── §5.3 step 5 — requests and negation ───────────────────────────────────────

def handle_requests_and_negation(tagged_df: pd.DataFrame, sentences_df: pd.DataFrame) -> pd.DataFrame:
    """
    §5.3 Step 5 — Negation and request handling.

    A sentence matching a request pattern ("wish it had a qibla") must not be
    scored as negative sentiment toward an existing qibla feature; it is
    flagged `is_request` and routed to 03c demand mining instead.
    """
    section("§5.3.5  Request and negation handling")

    sent_map = sentences_df.set_index(["reviewId", "sentence_index"])["sentence"]
    idx = pd.MultiIndex.from_arrays([tagged_df["reviewId"], tagged_df["sentence_index"]])
    tagged_df = tagged_df.copy()
    tagged_df["triggering_sentence"] = sent_map.reindex(idx).to_numpy()

    text = tagged_df["triggering_sentence"].fillna("")
    tagged_df["is_request"] = text.str.contains(REQUEST_RE, regex=True, na=False)
    tagged_df["has_negation"] = text.str.contains(NEGATION_RE, regex=True, na=False)

    log(f"Flagged as requests: {tagged_df['is_request'].sum():,} "
        f"({tagged_df['is_request'].mean():.1%}) — excluded from aspect sentiment, "
        f"routed to 03c.")
    log(f"Containing negation: {tagged_df['has_negation'].sum():,} "
        f"({tagged_df['has_negation'].mean():.1%})")
    return tagged_df


# ─── §5.3 step 4 — per-aspect sentiment ────────────────────────────────────────

def per_aspect_sentiment(
    tagged_df: pd.DataFrame, *, device: str, batch_size: int, fp32: bool,
    ckpt: Checkpoint,
) -> pd.DataFrame:
    """
    §5.3 Step 4 — Per-aspect sentiment.

    Scores each triggering sentence, then aggregates to (reviewId, aspect) by
    probability-weighted mean. Weighting by confidence rather than taking a
    plain mean keeps a hedged sentence from cancelling a decisive one.

    Sentences are deduplicated before scoring — the same short sentence recurs
    constantly across a 732K-review corpus.
    """
    section("§5.3.4  Per-aspect sentiment")

    scorable = tagged_df[~tagged_df["is_request"]].copy()
    log(f"Scoring {len(scorable):,} non-request (aspect, sentence) pairs.")

    uniq = pd.Index(scorable["triggering_sentence"].dropna().unique())
    log(f"Unique sentences to score: {len(uniq):,} "
        f"({1 - len(uniq) / max(1, len(scorable)):.0%} collapsed)")

    frames, clf = [], None
    shards = list(range(0, len(uniq), SHARD_SIZE))
    for shard_no, start in enumerate(shards):
        shard_id = f"sent_{shard_no:04d}"
        if ckpt.has(shard_id):
            frames.append(ckpt.read(shard_id))
            log(f"  shard {shard_id}: resumed")
            continue
        if clf is None:
            clf = SequenceClassifier(
                model_name="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                device=device, batch_size=batch_size, fp32=fp32, max_length=128,
            ).load()
        chunk = uniq[start: start + SHARD_SIZE].tolist()
        probs = clf.predict_proba(chunk, desc=f"aspect-sent {shard_no + 1}/{len(shards)}")
        labels = [lab.lower() for lab in clf.labels]
        neg_i = next(i for i, l in enumerate(labels) if "neg" in l)
        neu_i = next(i for i, l in enumerate(labels) if "neu" in l)
        pos_i = next(i for i, l in enumerate(labels) if "pos" in l)
        frame = pd.DataFrame({
            "triggering_sentence": chunk,
            "s_neg": probs[:, neg_i], "s_neu": probs[:, neu_i], "s_pos": probs[:, pos_i],
        })
        ckpt.write(shard_id, frame)
        frames.append(frame)

    scores = pd.concat(frames, ignore_index=True).drop_duplicates("triggering_sentence")
    scorable = scorable.merge(scores, on="triggering_sentence", how="left")

    # Probability-weighted aggregation: confidence = 1 - neutral mass.
    scorable["weight"] = (1.0 - scorable["s_neu"]).clip(lower=0.05)
    for c in ("s_neg", "s_neu", "s_pos"):
        scorable[f"w_{c}"] = scorable[c] * scorable["weight"]

    grouped = scorable.groupby(["reviewId", "aspect"], observed=True)
    agg = grouped.agg(
        prob_neg=("w_s_neg", "sum"),
        prob_neu=("w_s_neu", "sum"),
        prob_pos=("w_s_pos", "sum"),
        _w=("weight", "sum"),
        n_sentences=("triggering_sentence", "size"),
        triggering_sentence=("triggering_sentence", lambda s: " || ".join(s.astype(str)[:3])),
        match_method=("match_method", "first"),
        has_negation=("has_negation", "any"),
    ).reset_index()

    for c in ("prob_neg", "prob_neu", "prob_pos"):
        agg[c] = (agg[c] / agg["_w"]).astype("float32")
    agg = agg.drop(columns="_w")

    probs = agg[["prob_neg", "prob_neu", "prob_pos"]].to_numpy()
    agg["sentiment_label"] = np.array(["Negative", "Neutral", "Positive"])[probs.argmax(axis=1)]
    agg["is_request"] = False

    log(f"Aggregated to {len(agg):,} (reviewId, aspect) rows.")
    dist = agg["sentiment_label"].value_counts(normalize=True)
    log("Aspect sentiment distribution:")
    for lab in ("Positive", "Neutral", "Negative"):
        log(f"    {lab:9s} {dist.get(lab, 0.0):.1%}")
    return agg


def _request_rows(tagged_df: pd.DataFrame) -> pd.DataFrame:
    """Request-flagged pairs, carried through unscored for 03c."""
    req = tagged_df[tagged_df["is_request"]]
    if req.empty:
        return pd.DataFrame()
    out = (
        req.groupby(["reviewId", "aspect"], observed=True)
           .agg(
               triggering_sentence=("triggering_sentence", lambda s: " || ".join(s.astype(str)[:3])),
               match_method=("match_method", "first"),
               has_negation=("has_negation", "any"),
               n_sentences=("triggering_sentence", "size"),
           )
           .reset_index()
    )
    out["prob_neg"] = np.nan
    out["prob_neu"] = np.nan
    out["prob_pos"] = np.nan
    out["sentiment_label"] = pd.NA
    out["is_request"] = True
    return out


# ─── Pipeline ───────────────────────────────────────────────────────────────────

def main() -> None:
    p = base_parser(__doc__ or "Phase 2 — ABSA")
    p.add_argument("--no-zero-shot", action="store_true",
                   help="Skip the zero-shot backstop (lexicon matching only).")
    p.add_argument("--zeroshot-threshold", type=float, default=0.75,
                   help="Entailment probability required to accept a zero-shot aspect.")
    p.add_argument("--zeroshot-topk", type=int, default=5,
                   help="Candidate labels per sentence from the embedding prefilter. "
                        "0 disables the prefilter and scores all 27 (much slower).")
    p.add_argument("--zeroshot-max", type=int, default=None, metavar="N",
                   help="Cap zero-shot to N unmatched sentences.")
    p.add_argument("--fast-split", action="store_true",
                   help="Use the regex sentence splitter instead of pysbd.")
    args = p.parse_args()
    set_seed(args.seed)

    require(SENTIMENT_PATH, "python src/scripts/02b_sentiment_roberta.py")
    df = pd.read_parquet(
        SENTIMENT_PATH,
        columns=["reviewId", "app_name", "content_clean", "lang", "score", "corpus_text"],
    )
    df = df[df["corpus_text"]].copy()
    log(f"C_text subset: {len(df):,} reviews")
    df = maybe_sample(df, args)

    device, vram = get_device(args.device)
    lexicon = load_aspect_lexicon()

    sentences = segment_sentences(df, fast=args.fast_split)
    tagged = lexicon_tagging(sentences, lexicon)

    if not args.no_zero_shot:
        zs_batch = auto_batch_size(vram, "large", requested=args.batch_size)
        zs = zero_shot_backstop(
            sentences, tagged, device=device, batch_size=zs_batch, fp32=args.fp32,
            threshold=args.zeroshot_threshold,
            top_k=args.zeroshot_topk or len(ASPECT_DESCRIPTIONS),
            max_sentences=args.zeroshot_max,
        )
        if len(zs):
            tagged = pd.concat([tagged, zs], ignore_index=True)
    else:
        log("Zero-shot backstop disabled (--no-zero-shot).")

    tagged = handle_requests_and_negation(tagged, sentences)

    sent_batch = auto_batch_size(vram, "base", requested=args.batch_size)
    ckpt = Checkpoint("absa", resume=args.resume, overwrite=args.overwrite)
    scored = per_aspect_sentiment(
        tagged, device=device, batch_size=sent_batch, fp32=args.fp32, ckpt=ckpt,
    )

    out = pd.concat([scored, _request_rows(tagged)], ignore_index=True)
    out = out.merge(df[["reviewId", "app_name", "score", "lang"]], on="reviewId", how="left")
    out["aspect_group"] = np.where(
        out["aspect"].isin(CSV_LINKED_ASPECTS), "csv_linked", "review_only"
    )

    section("Writing output")
    out.to_parquet(ASPECT_PATH, index=False, compression="zstd")
    summarize(out, "aspect_sentiments")
    log(f"Saved → {ASPECT_PATH}")
    log("")
    log(f"Reviews with >=1 aspect: {out['reviewId'].nunique():,}")
    log(f"Request-flagged rows:    {int(out['is_request'].sum()):,}")
    log("")
    log("Next: python src/scripts/03b_promise_extraction.py")


if __name__ == "__main__":
    main()
