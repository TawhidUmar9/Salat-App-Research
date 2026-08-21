"""
Phase 1B — Transformer Sentiment Classification
================================================
CardiffNLP RoBERTa for English, XLM-RoBERTa for everything else.

Input:
    - data/master_reviews.parquet (corpus_text subset)
    - data/_checkpoints/vader/vader.parquet (from 02a)

Output:
    - data/master_reviews_with_sentiment.parquet (all sentiment columns merged)

See implementation_plan.md §4.2, §4.3 for full specification.

DESIGN NOTE — deviation from the plan, deliberate:
    §3.2 proposed machine-translating non-English reviews before scoring.
    Instead, English rows go through `twitter-roberta-base-sentiment-latest`
    and non-English rows through `twitter-xlm-roberta-base-sentiment`, which
    scores Bengali/Arabic/Urdu natively. This avoids ~100K translation calls
    and the MT noise §16.3 warns about. Which model scored a row is recorded
    in `sentiment_model`, so C_en and C_trans results stay separable — the
    plan's requirement that translated results never be pooled into headline
    numbers is preserved.

Usage:
    python src/scripts/02b_sentiment_roberta.py
    python src/scripts/02b_sentiment_roberta.py --sample 20000
    python src/scripts/02b_sentiment_roberta.py --resume
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CKPT_DIR, MASTER_PATH, SENTIMENT_PATH,
    Checkpoint, auto_batch_size, base_parser, get_device, log, maybe_sample,
    require, section, set_seed, summarize,
)
from _nlp import SequenceClassifier  # noqa: E402

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MODEL_NAME_MULTI = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

VADER_PATH = CKPT_DIR / "vader" / "vader.parquet"

#: Rows per checkpoint shard. A crash costs at most this much work.
SHARD_SIZE = 50_000


def _canonical_probs(probs: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """
    Map a model's own label order onto (neg, neu, pos).

    twitter-roberta-base-sentiment-latest emits negative/neutral/positive;
    the XLM variant emits Negative/Neutral/Positive. Neither order is
    guaranteed by the checkpoint, so resolve it from config.id2label.
    """
    idx = {}
    for i, lab in enumerate(labels):
        low = lab.lower()
        if "neg" in low or low == "label_0":
            idx["neg"] = i
        elif "neu" in low or low == "label_1":
            idx["neu"] = i
        elif "pos" in low or low == "label_2":
            idx["pos"] = i
    missing = {"neg", "neu", "pos"} - set(idx)
    if missing:
        raise ValueError(f"Cannot map sentiment labels {labels}: missing {missing}")
    return pd.DataFrame({
        "roberta_neg": probs[:, idx["neg"]],
        "roberta_neu": probs[:, idx["neu"]],
        "roberta_pos": probs[:, idx["pos"]],
    })


def run_roberta_sentiment(
    df: pd.DataFrame, *, device: str, batch_size: int, fp32: bool,
    ckpt: Checkpoint, multilingual: bool = True,
) -> pd.DataFrame:
    """
    §4.2 — Transformer sentiment, batched with checkpoint/resume.

    Stores all three class probabilities, not just the argmax: Phase 2
    aggregates aspect sentiment by probability-weighted mean and needs the
    full mass.
    """
    section("§4.2  Transformer sentiment")

    lang = df["lang"].fillna("unk")
    groups: list[tuple[str, str, pd.Index]] = [("en", MODEL_NAME, df.index[lang.eq("en")])]
    if multilingual:
        groups.append(("multi", MODEL_NAME_MULTI, df.index[~lang.eq("en")]))
    else:
        groups[0] = ("en", MODEL_NAME, df.index)

    results = []
    for tag, model_name, idx in groups:
        if len(idx) == 0:
            continue
        log(f"{tag}: {len(idx):,} reviews → {model_name}")

        clf = None
        shard_frames = []
        shards = list(range(0, len(idx), SHARD_SIZE))

        for shard_no, start in enumerate(shards):
            shard_id = f"{tag}_{shard_no:04d}"
            if ckpt.has(shard_id):
                shard_frames.append(ckpt.read(shard_id))
                log(f"  shard {shard_id}: resumed from checkpoint")
                continue

            if clf is None:   # lazy: skip model load entirely on a full resume
                clf = SequenceClassifier(
                    model_name=model_name, device=device,
                    batch_size=batch_size, fp32=fp32,
                ).load()

            sub = df.loc[idx[start: start + SHARD_SIZE]]
            probs = clf.predict_proba(
                sub["content_clean"].fillna("").tolist(),
                desc=f"{tag} {shard_no + 1}/{len(shards)}",
            )
            out = _canonical_probs(probs, clf.labels)
            out.insert(0, "reviewId", sub["reviewId"].values)
            out["sentiment_model"] = tag
            ckpt.write(shard_id, out)
            shard_frames.append(out)

        results.append(pd.concat(shard_frames, ignore_index=True))
        del clf

    scored = pd.concat(results, ignore_index=True)
    probs = scored[["roberta_neg", "roberta_neu", "roberta_pos"]].to_numpy()
    scored["roberta_label"] = np.array(["Negative", "Neutral", "Positive"])[probs.argmax(axis=1)]
    scored["roberta_confidence"] = probs.max(axis=1)

    dist = scored["roberta_label"].value_counts(normalize=True)
    log("Transformer label distribution:")
    for lab in ("Positive", "Neutral", "Negative"):
        log(f"    {lab:9s} {dist.get(lab, 0.0):.1%}")
    return scored


def add_star_proxy_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    §4.3 — Star-rating proxy sentiment (Method C).

    KNOWN LIMITATION, and central to RQ2: the star is an app-level verdict,
    so a 5-star review saying "only complaint is the qibla is off" is
    mislabelled at the aspect level. That divergence is the phenomenon RQ2
    exploits — it is measured in 02c, not assumed away.
    """
    section("§4.3  Star-rating proxy")

    score = pd.to_numeric(df["score"], errors="coerce")
    df["star_label"] = np.select(
        [score <= 2, score == 3, score >= 4],
        ["Negative", "Neutral", "Positive"],
        default=None,
    )
    dist = pd.Series(df["star_label"]).value_counts(normalize=True, dropna=True)
    log("Star-proxy distribution:")
    for lab in ("Positive", "Neutral", "Negative"):
        log(f"    {lab:9s} {dist.get(lab, 0.0):.1%}")
    return df


def merge_all_sentiment(master: pd.DataFrame, roberta: pd.DataFrame) -> pd.DataFrame:
    """Merge VADER (02a) + transformer + star proxy onto the master frame."""
    section("Merging sentiment sources")

    out = master.merge(roberta, on="reviewId", how="left", validate="one_to_one")

    if VADER_PATH.exists():
        vader = pd.read_parquet(VADER_PATH)
        out = out.merge(vader, on="reviewId", how="left", validate="one_to_one")
        log(f"Merged VADER for {out['vader_label'].notna().sum():,} reviews.")
    else:
        log("VADER output not found — run 02a for the three-way comparison in 02c.", level="WARN")
        for c in ["vader_neg", "vader_neu", "vader_pos", "vader_compound"]:
            out[c] = np.nan
        out["vader_label"] = pd.NA
        out["vader_applicable"] = False

    out = add_star_proxy_sentiment(out)

    scored = out["roberta_label"].notna()
    log(f"Reviews with transformer sentiment: {scored.sum():,} / {len(out):,}")
    return out


def main() -> None:
    p = base_parser(__doc__ or "Phase 1B — transformer sentiment")
    p.add_argument(
        "--english-only", action="store_true",
        help="Score only English rows (skips the multilingual model entirely).",
    )
    args = p.parse_args()
    set_seed(args.seed)

    require(MASTER_PATH, "python src/scripts/01_preprocess.py")
    master = pd.read_parquet(MASTER_PATH)
    log(f"Loaded master: {len(master):,} reviews")

    df_text = master[master["corpus_text"]].copy()
    log(f"C_text subset: {len(df_text):,} reviews")
    df_text = maybe_sample(df_text, args)

    device, vram = get_device(args.device)
    batch_size = auto_batch_size(vram, "base", requested=args.batch_size)
    ckpt = Checkpoint("roberta", resume=args.resume, overwrite=args.overwrite)

    scored = run_roberta_sentiment(
        df_text, device=device, batch_size=batch_size, fp32=args.fp32,
        ckpt=ckpt, multilingual=not args.english_only,
    )

    out = merge_all_sentiment(master, scored)

    section("Writing output")
    out.to_parquet(SENTIMENT_PATH, index=False, compression="zstd")
    summarize(out, "master_reviews_with_sentiment")
    log(f"Saved → {SENTIMENT_PATH}")

    log("")
    log("Next: notebooks/02c_cross_validation.py, then python src/scripts/03_absa.py")


if __name__ == "__main__":
    main()
