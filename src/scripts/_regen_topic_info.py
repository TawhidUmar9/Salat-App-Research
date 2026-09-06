"""
Rebuild `topic_info.csv` for the ecosystem model — recovery, not analysis.
=========================================================================
`label_topics` used to hardcode its output path, so every sub-model in
`04_topic_modeling.py` overwrote the 40-topic ecosystem sheet on its way past.
Runs finished with `topic_info.csv` holding the LAST sub-model (`rq5_bloat`)
instead of the ecosystem topics. The bug is fixed; this recovers the sheet from
a run that predates the fix, without refitting.

Nothing is lost, because the ecosystem model is saved and the assignments are in
topics.parquet:
    - keywords / auto_name  ← the saved BERTopic model's c-TF-IDF
    - sizes                 ← topics.parquet (the actual assignment record)
    - representative docs   ← topics.parquet joined to the reviews

Representative docs are NOT byte-identical to what BERTopic picked: the saved
model does not serialize `representative_docs_`. They are chosen here by highest
topic_probability, falling back to longest-document when probabilities were not
computed. Both are deterministic. Read the docs in `04_topic_exploration.py`
anyway — it shows 15 per topic against these 2.

Usage:
    python src/scripts/_regen_topic_info.py
    python src/scripts/_regen_topic_info.py --force   # overwrite filled labels
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    DATA_DIR, SENTIMENT_PATH, TOPICS_PATH, log, require, section,
)

from importlib import import_module  # noqa: E402

# Module name starts with a digit, so it cannot be imported with `import`.
RQ_KEYWORDS = import_module("04_topic_modeling").RQ_KEYWORDS

OUT_PATH = DATA_DIR / "topic_info.csv"
MODEL_DIR = DATA_DIR / "_models" / "bertopic"
DOC_CHARS = 300


def _guard_existing(force: bool) -> None:
    """Never silently destroy hand-written labels."""
    if not OUT_PATH.exists():
        return
    existing = pd.read_csv(OUT_PATH)
    filled = existing.get("manual_label", pd.Series(dtype=str)).fillna("").str.strip()
    n = int((filled != "").sum())
    if n and not force:
        raise SystemExit(
            f"\n{OUT_PATH} already has {n} manual labels filled in.\n"
            f"Refusing to overwrite. Back it up, then re-run with --force.\n"
        )
    if n:
        log(f"--force: discarding {n} existing manual labels", level="WARN")


def main() -> None:
    force = "--force" in sys.argv
    section("Rebuild ecosystem topic_info.csv")
    _guard_existing(force)

    require(TOPICS_PATH, "src/scripts/04_topic_modeling.py")
    require(SENTIMENT_PATH, "src/scripts/02b_sentiment_roberta.py")
    require(MODEL_DIR, "src/scripts/04_topic_modeling.py")

    from bertopic import BERTopic
    model = BERTopic.load(str(MODEL_DIR))

    topics = pd.read_parquet(TOPICS_PATH)
    if "topic_id" not in topics.columns:
        raise SystemExit(f"{TOPICS_PATH} has no topic_id column: {list(topics.columns)}")

    reviews = pd.read_parquet(SENTIMENT_PATH, columns=["reviewId", "content_clean"])
    merged = topics.merge(reviews, on="reviewId", how="left")

    has_prob = ("topic_probability" in merged.columns
                and merged["topic_probability"].notna().any())
    log(f"{len(merged):,} assigned documents; "
        f"representative docs by {'topic_probability' if has_prob else 'document length'}")

    merged["_len"] = merged["content_clean"].fillna("").str.len()
    sort_cols = (["topic_probability", "_len"] if has_prob else ["_len"])
    merged = merged.sort_values(sort_cols + ["reviewId"],
                                ascending=[False] * len(sort_cols) + [True])

    info = model.get_topic_info().set_index("Topic")

    rows = []
    for tid, grp in merged.groupby("topic_id", sort=False):
        words = [w for w, _ in (model.get_topic(int(tid)) or [])][:10]
        blob = " ".join(words).lower()
        docs = grp["content_clean"].fillna("").tolist()
        rows.append({
            "topic_id": int(tid),
            "size": int(len(grp)),
            "auto_name": info["Name"].get(int(tid), ""),
            "top_words": ", ".join(words),
            "rq_tags": "|".join(rq for rq, kws in RQ_KEYWORDS.items()
                                if any(k in blob for k in kws)),
            "representative_doc_1": docs[0][:DOC_CHARS] if docs else "",
            "representative_doc_2": docs[1][:DOC_CHARS] if len(docs) > 1 else "",
            "manual_label": "",
        })

    out = pd.DataFrame(rows).sort_values("size", ascending=False)
    n_topics = int((out["topic_id"] >= 0).sum())
    log(f"Rebuilt {n_topics} topics (+ outlier bucket)")
    if n_topics < 20:
        log(f"Only {n_topics} topics — this does not look like the ecosystem "
            f"model. Check that {MODEL_DIR} is from the full run.", level="WARN")

    log(f"{'id':>4s} {'n':>7s}  {'RQ':10s} top words")
    for _, r in out.head(40).iterrows():
        log(f"    {r['topic_id']:>4d} {r['size']:>7,}  {r['rq_tags']:10s} {r['top_words'][:70]}")

    out.to_csv(OUT_PATH, index=False)
    log(f"Saved → {OUT_PATH}")
    log("ACTION: fill `manual_label` from the representative documents.")


if __name__ == "__main__":
    main()
