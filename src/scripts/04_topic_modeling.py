"""
Phase 3 — Topic Modelling
==========================
BERTopic with sentence-transformers on the analysis sub-corpus.

Input:
    - data/master_reviews_with_sentiment.parquet (C_en subset)

Output:
    - data/topics.parquet (reviewId, topic_id, topic_label, topic_probability)
    - data/topic_info.csv (topic sizes, keywords, RQ tags)

See implementation_plan.md §7 for full specification.

Usage:
    python src/scripts/04_topic_modeling.py
    python src/scripts/04_topic_modeling.py --sample 30000
    python src/scripts/04_topic_modeling.py --nr-topics 20
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    ASPECT_PATH, DATA_DIR, SENTIMENT_PATH, TOPICS_PATH,
    base_parser, get_device, log, maybe_sample, record_run_fact, require, section,
    set_seed, summarize,
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_MODEL_MULTI = "paraphrase-multilingual-MiniLM-L12-v2"

#: Domain stopwords (§7). Without these, every cluster is "allah good app muslim"
#: and the model resolves nothing. App names are added dynamically in main().
DOMAIN_STOPWORDS = [
    "allah", "islam", "islamic", "muslim", "muslims", "app", "apps", "application",
    "good", "nice", "great", "best", "very", "really", "thanks", "thank",
    "alhamdulillah", "mashallah", "masha", "inshallah", "jazakallah", "subhanallah",
    "please", "download", "downloaded", "use", "using", "used", "make", "made",
    "like", "love", "just", "get", "got", "one", "also", "well", "much", "many",
    "prayer", "prayers", "pray", "praying", "salah", "salat", "namaz", "time", "times",
]

#: Topic-label keyword → RQ mapping, used to tag topics for the deep-dive notebooks.
RQ_KEYWORDS = {
    "RQ1": ["streak", "track", "score", "goal", "badge", "progress", "statistic", "habit"],
    "RQ2": ["time", "wrong", "accurate", "qibla", "compass", "direction", "calculation",
            "madhab", "hanafi", "ui", "design", "interface", "layout"],
    "RQ3": ["travel", "qasr", "journey", "timezone", "mosque", "masjid", "nearby"],
    "RQ4": ["women", "period", "menstruation", "privacy", "data", "permission"],
    "RQ5": ["feature", "cluttered", "complicated", "simple", "widget", "watch",
            "ring", "wearable", "crash", "bug", "slow"],
    "RQ6": ["missing", "wish", "add", "need", "lack", "premium", "subscription", "ads"],
}


def _make_vectorizer(extra_stopwords: list[str]):
    from sklearn.feature_extraction.text import CountVectorizer
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        base = list(ENGLISH_STOP_WORDS)
    except ImportError:
        base = []
    stops = sorted(set(base) | set(DOMAIN_STOPWORDS) | set(extra_stopwords))
    # NOTE ON min_df / max_df: BERTopic applies this vectorizer to *topic-
    # aggregated* documents during c-TF-IDF — one concatenated document per
    # topic — not to the raw reviews. So `min_df=5` would mean "term must occur
    # in at least 5 of the ~20 topics", which deletes exactly the distinctive
    # terms that name a topic, and `max_df=0.5` crashes outright with
    # "max_df corresponds to < documents than min_df" whenever the topic count
    # drops below 10. Filtering here is the stopword list's job; leave the
    # frequency bounds permissive.
    return CountVectorizer(
        stop_words=stops, ngram_range=(1, 2), min_df=1, max_df=1.0,
    )


def _app_name_stopwords(df: pd.DataFrame) -> list[str]:
    """Tokens from app names — otherwise each app forms its own topic."""
    tokens = set()
    for name in df["app_name"].dropna().unique():
        for tok in str(name).lower().replace(":", " ").replace(",", " ").split():
            if len(tok) > 2:
                tokens.add(tok)
    return sorted(tokens)


def build_topic_model(docs: list[str], *, device: str, extra_stopwords: list[str],
                      nr_topics: int | str = "auto", min_topic_size: int | None = None,
                      multilingual: bool = False, seed: int = 42):
    """
    §7 — Fit BERTopic on C_en.

    min_topic_size scales with corpus size: too small and the model produces
    hundreds of near-duplicate micro-topics, too large and distinct complaints
    collapse together. ~0.1% of the corpus, floored at 50, works well here.
    """
    section("§7  Fitting BERTopic")

    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    if min_topic_size is None:
        min_topic_size = max(50, int(len(docs) * 0.001))
    log(f"{len(docs):,} documents; min_topic_size={min_topic_size}; nr_topics={nr_topics}")

    # UMAP + HDBSCAN run on CPU and are the pipeline's memory ceiling: UMAP's
    # k-NN graph is superlinear in document count, so this stage — not the
    # transformers — is what fails first at corpus scale. Say so before spending
    # the embedding time, and record the size actually fitted so the paper can
    # state it if a subsample was used.
    record_run_fact("topic_model_docs", len(docs))
    record_run_fact("topic_model_min_topic_size", min_topic_size)
    if len(docs) > 150_000:
        log(f"{len(docs):,} documents is large for UMAP/HDBSCAN (CPU-bound, "
            f"superlinear memory). If this is killed or thrashes, re-run with "
            f"--sample N and REPORT the subsample size in the paper.", level="WARN")

    model_name = EMBEDDING_MODEL_MULTI if multilingual else EMBEDDING_MODEL
    embedder = SentenceTransformer(model_name, device=device)
    log(f"Embedding with {model_name} on {device}...")
    embeddings = embedder.encode(
        docs, batch_size=256, show_progress_bar=True, convert_to_numpy=True,
    )

    # BERTopic's default UMAP is UNSEEDED, and UMAP is stochastic: two runs over
    # the identical 276,422 documents produced 38 topics (30.6% unassigned) and
    # then 74 topics (20.9%). A topic count that swings by 2x between runs cannot
    # be reported, and every downstream label and quote would move with it.
    #
    # Seeding costs speed — UMAP falls back to single-threaded when random_state
    # is set — but a reproducible model is not optional for a paper. The other
    # parameters are UMAP's defaults, restated so the seed is not the only thing
    # pinned.
    from umap import UMAP
    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
        random_state=seed,
    )
    log(f"UMAP seeded with random_state={seed} (single-threaded, slower but "
        f"reproducible).")

    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=umap_model,
        vectorizer_model=_make_vectorizer(extra_stopwords),
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        calculate_probabilities=False,
        verbose=True,
    )
    topics, probs = topic_model.fit_transform(docs, embeddings)

    info = topic_model.get_topic_info()
    n_outlier = int((np.array(topics) == -1).sum())
    log(f"Fitted {len(info) - 1} topics; {n_outlier:,} documents unassigned "
        f"({n_outlier / len(docs):.1%})")
    return topic_model, topics, probs


def label_topics(topic_model, docs: list[str]) -> pd.DataFrame:
    """
    §7 — Label topics from representative documents and tag them by RQ.

    BERTopic's default c-TF-IDF names are keyword salads; the representative
    documents are what a human actually reads to name a topic, so both are
    written out for the manual labelling pass.
    """
    section("§7  Topic labelling")

    info = topic_model.get_topic_info()
    reps = topic_model.get_representative_docs()

    rows = []
    for _, r in info.iterrows():
        tid = int(r["Topic"])
        words = [w for w, _ in (topic_model.get_topic(tid) or [])][:10]
        blob = " ".join(words).lower()
        rqs = [rq for rq, kws in RQ_KEYWORDS.items() if any(k in blob for k in kws)]
        rows.append({
            "topic_id": tid,
            "size": int(r["Count"]),
            "auto_name": r.get("Name", ""),
            "top_words": ", ".join(words),
            "rq_tags": "|".join(rqs),
            "representative_doc_1": (reps.get(tid) or [""])[0][:300],
            "representative_doc_2": (reps.get(tid) or ["", ""])[1][:300] if len(reps.get(tid) or []) > 1 else "",
            "manual_label": "",       # ACTION: fill during the labelling pass
        })

    out = pd.DataFrame(rows).sort_values("size", ascending=False)
    log(f"{'id':>4s} {'n':>7s}  {'RQ':10s} top words")
    for _, r in out.head(30).iterrows():
        log(f"    {r['topic_id']:>4d} {r['size']:>7,}  {r['rq_tags']:10s} {r['top_words'][:70]}")

    path = DATA_DIR / "topic_info.csv"
    out.to_csv(path, index=False)
    log(f"Saved → {path}")
    log("ACTION: fill `manual_label` in topic_info.csv from the representative docs.")
    return out


def run_subtopic_models(df: pd.DataFrame, *, device: str, extra_stopwords: list[str],
                        seed: int = 42) -> dict:
    """
    §7 — Sub-topic models within specific aspects.

    The ecosystem-level model cannot resolve streak-anxiety from
    streak-motivation: both are a small fraction of a 300K corpus and merge
    into one "tracker" topic. Fitting within the tracker subset is what makes
    RQ1's love/hate split visible.
    """
    section("§7  Sub-topic models (RQ1 tracker, RQ4 women/privacy, RQ5 bloat)")

    if not ASPECT_PATH.exists():
        log("aspect_sentiments.parquet not found — skipping sub-topic models.", level="WARN")
        return {}

    asp = pd.read_parquet(ASPECT_PATH, columns=["reviewId", "aspect", "is_request"])
    subsets = {
        "rq1_tracker": ["prayer_tracker", "tracker_score", "goal_system"],
        "rq4_women_privacy": ["women_period", "privacy_data"],
        # RQ5 reframed: feature_count does not predict bloat complaints, so the
        # question becomes what those complaints are actually ABOUT. Fitted on
        # complexity_bloat alone — deliberately NOT pooled with ui_design, which
        # would assume the surfacing conclusion instead of testing for it.
        "rq5_bloat": ["complexity_bloat"],
    }

    results = {}
    for name, aspects in subsets.items():
        ids = set(asp.loc[asp["aspect"].isin(aspects), "reviewId"])
        sub = df[df["reviewId"].isin(ids)]
        log(f"{name}: {len(sub):,} reviews")
        if len(sub) < 200:
            log(f"  too few reviews for a stable sub-model — reporting count only.", level="WARN")
            results[name] = None
            continue

        model, topics, _ = build_topic_model(
            sub["content_clean"].tolist(), device=device,
            extra_stopwords=extra_stopwords, seed=seed,
            nr_topics=min(12, max(4, len(sub) // 400)),
            min_topic_size=max(15, len(sub) // 100),
        )
        info = label_topics(model, sub["content_clean"].tolist())
        info["submodel"] = name
        path = DATA_DIR / f"topic_info_{name}.csv"
        info.to_csv(path, index=False)
        log(f"Saved → {path}")
        results[name] = pd.DataFrame({"reviewId": sub["reviewId"].values, "topic_id": topics})

    return results


def topic_prevalence(topics_df: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """Topic × app prevalence and topic × sentiment, for Figure 4 and the deep dives."""
    section("§7  Prevalence and sentiment crosstabs")

    merged = topics_df.merge(
        reviews[["reviewId", "app_name", "roberta_label", "score"]], on="reviewId", how="left"
    )
    assigned = merged[merged["topic_id"] >= 0]

    prevalence = (
        assigned.groupby(["app_name", "topic_id"], observed=True).size()
        .rename("n").reset_index()
    )
    prevalence["share_of_app"] = prevalence["n"] / prevalence.groupby("app_name", observed=True)["n"].transform("sum")

    sentiment = (
        assigned.groupby(["topic_id", "roberta_label"], observed=True).size()
        .unstack(fill_value=0)
    )
    sentiment = sentiment.div(sentiment.sum(axis=1), axis=0)

    prevalence.to_csv(DATA_DIR / "topic_prevalence_by_app.csv", index=False)
    sentiment.to_csv(DATA_DIR / "topic_sentiment.csv")
    log(f"Saved topic_prevalence_by_app.csv and topic_sentiment.csv")

    if "Negative" in sentiment.columns:
        worst = sentiment["Negative"].sort_values(ascending=False).head(10)
        log("Most negative topics:")
        for tid, frac in worst.items():
            log(f"    topic {tid:>4d}  {frac:.0%} negative")
    return prevalence


def main() -> None:
    p = base_parser(__doc__ or "Phase 3 — topic modelling")
    p.add_argument("--nr-topics", default="auto",
                   help="Target topic count after reduction, or 'auto'. §7 recommends 15–25.")
    p.add_argument("--min-topic-size", type=int, default=None,
                   help="Minimum documents per topic. Default: 0.1%% of the corpus, floor 50.")
    p.add_argument("--multilingual", action="store_true",
                   help="Fit on C_text (all languages) with the multilingual encoder "
                        "instead of C_en. Report separately; never pool (§16.3).")
    p.add_argument("--skip-subtopics", action="store_true")
    args = p.parse_args()
    set_seed(args.seed)

    require(SENTIMENT_PATH, "python src/scripts/02b_sentiment_roberta.py")
    df = pd.read_parquet(
        SENTIMENT_PATH,
        columns=["reviewId", "app_name", "content_clean", "roberta_label", "score",
                 "corpus_en", "corpus_text"],
    )

    corpus_col = "corpus_text" if args.multilingual else "corpus_en"
    df = df[df[corpus_col]].copy()
    log(f"{corpus_col} subset: {len(df):,} reviews")
    if len(df) < 500:
        raise SystemExit("Too few documents to fit a topic model. Try a larger --sample.")
    df = maybe_sample(df, args)

    device, _ = get_device(args.device)
    extra_stops = _app_name_stopwords(df)
    log(f"App-name stopwords: {len(extra_stops)} tokens")

    nr_topics = args.nr_topics if args.nr_topics == "auto" else int(args.nr_topics)
    model, topics, probs = build_topic_model(
        df["content_clean"].tolist(), device=device, extra_stopwords=extra_stops,
        nr_topics=nr_topics, min_topic_size=args.min_topic_size,
        multilingual=args.multilingual, seed=args.seed,
    )

    label_topics(model, df["content_clean"].tolist())

    topics_df = pd.DataFrame({
        "reviewId": df["reviewId"].values,
        "topic_id": np.asarray(topics, dtype="int32"),
        "topic_probability": (np.asarray(probs, dtype="float32")
                              if probs is not None else np.nan),
    })
    topic_prevalence(topics_df, df)

    if not args.skip_subtopics:
        subs = run_subtopic_models(df, device=device, extra_stopwords=extra_stops,
                                   seed=args.seed)
        for name, sub_df in subs.items():
            if sub_df is None:
                continue
            topics_df = topics_df.merge(
                sub_df.rename(columns={"topic_id": f"{name}_topic"}),
                on="reviewId", how="left",
            )

    section("Writing output")
    topics_df.to_parquet(TOPICS_PATH, index=False, compression="zstd")
    summarize(topics_df, "topics")
    log(f"Saved → {TOPICS_PATH}")

    model_dir = DATA_DIR / "_models" / "bertopic"
    try:
        model.save(str(model_dir), serialization="safetensors", save_ctfidf=True)
        log(f"Saved fitted model → {model_dir}")
    except Exception as exc:  # noqa: BLE001
        log(f"Could not serialize BERTopic model: {exc}", level="WARN")

    log("")
    log("Next: python src/scripts/05_temporal.py")


if __name__ == "__main__":
    main()
