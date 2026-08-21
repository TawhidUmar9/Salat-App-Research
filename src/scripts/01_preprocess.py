"""
Phase 0 — Preprocessing
========================
Merge, clean, detect language, and define analysis sub-corpora.

Input:
    - reviews/*/reviews.csv           (26 app review CSVs)
    - reviews/*/metadata.json         (26 app metadata files)
    - Salah App Analysis - Sheet1.csv (feature matrix, 20 annotated apps)

Output:
    - data/master_reviews.parquet     (all reviews with corpus_* boolean flags)

See implementation_plan.md §3 (Phase 0) for full specification.

Usage:
    python src/scripts/01_preprocess.py                    # full corpus
    python src/scripts/01_preprocess.py --sample 20000     # fast dev run
    python src/scripts/01_preprocess.py --translate-sample 500
"""

from __future__ import annotations

import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    MASTER_PATH, MODELS_DIR, REVIEWS_DIR,
    base_parser, load_feature_csv, load_metadata, log, maybe_sample,
    section, set_seed, summarize,
)

# ─── Configuration ──────────────────────────────────────────────────────────────

FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
FASTTEXT_PATH = MODELS_DIR / "lid.176.bin"

#: A review needs at least this many words to carry aspect content (§2.3).
MIN_WORDS_FOR_TEXT_CORPUS = 5

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
WHITESPACE_RE = re.compile(r"\s+")


def _is_emoji(ch: str) -> bool:
    """Emoji, pictographs, dingbats and regional indicators — not ordinary script."""
    cp = ord(ch)
    return (
        0x1F300 <= cp <= 0x1FAFF      # pictographs, emoticons, symbols, extended-A
        or 0x2600 <= cp <= 0x27BF     # misc symbols + dingbats
        or 0x1F1E6 <= cp <= 0x1F1FF   # regional indicators (flags)
        or cp in (0x2764, 0x2B50, 0xFE0F, 0x200D)
        or unicodedata.category(ch) == "So"
    )


EMOJI_TRANSLATION = None  # built lazily; see strip_emoji


def strip_emoji(text: str) -> tuple[str, str]:
    """Return (text_without_emoji, concatenated_emoji)."""
    if not text:
        return "", ""
    if not any(ord(c) > 0x2500 for c in text):   # fast path: no emoji possible
        return text, ""
    kept, found = [], []
    for ch in text:
        if _is_emoji(ch):
            found.append(ch)
        else:
            kept.append(ch)
    return "".join(kept), "".join(found)


# ─── §3.1 Merge & normalize ─────────────────────────────────────────────────────

def load_and_merge_reviews() -> pd.DataFrame:
    """
    §3.1 — Merge & normalize

    Concatenates all 26 reviews/*/reviews.csv, joins the app-level metadata
    fields the models use as covariates, and deduplicates on reviewId.
    """
    section("§3.1  Loading and merging review CSVs")

    meta = load_metadata()
    meta_by_dir = meta.set_index("app_dir")
    log(f"Metadata loaded for {len(meta)} apps.")

    frames = []
    for d in sorted(REVIEWS_DIR.iterdir()):
        csv_file = d / "reviews.csv"
        if not d.is_dir() or not csv_file.exists():
            continue
        # utf-8-sig: the scraper wrote a BOM ahead of the reviewId header.
        part = pd.read_csv(
            csv_file, encoding="utf-8-sig", dtype=str,
            on_bad_lines="warn", low_memory=False,
        )
        part["app_dir"] = d.name
        frames.append(part)
        log(f"  {d.name:38s} {len(part):>7,} reviews")

    if not frames:
        raise SystemExit(f"No reviews found under {REVIEWS_DIR}")

    df = pd.concat(frames, ignore_index=True)
    log(f"Concatenated: {len(df):,} rows from {len(frames)} apps.")

    before = len(df)
    df = df.drop_duplicates(subset="reviewId", keep="first")
    if before - len(df):
        log(f"Deduplicated on reviewId: removed {before - len(df):,} duplicates.")

    meta_cols = [
        "app_name", "package", "realInstalls", "app_score", "app_ratings",
        "released_dt", "adSupported", "offersIAP", "genre", "developer", "app_key",
    ]
    df = df.join(meta_by_dir[meta_cols], on="app_dir")

    df["score"] = pd.to_numeric(df["score"], errors="coerce").astype("Int8")
    df["thumbsUpCount"] = pd.to_numeric(df["thumbsUpCount"], errors="coerce").fillna(0).astype("int32")
    df["content"] = df["content"].fillna("")

    summarize(df, "Merged corpus")
    return df.reset_index(drop=True)


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    §3.1 — Text normalization

    Keeps raw `content` untouched. Produces `content_clean` (casing preserved,
    for the transformer), `content_lower` (for lexicon matching), and pulls
    emoji into their own column first — they are sentiment-bearing and
    disproportionately common in the short reviews.
    """
    section("§3.1  Text normalization")

    raw = df["content"].astype(str)

    stripped = raw.map(strip_emoji)
    df["emoji"] = stripped.map(lambda t: t[1])
    body = stripped.map(lambda t: t[0])

    body = body.str.replace(URL_RE, " ", regex=True)
    body = body.str.replace(EMAIL_RE, " ", regex=True)
    body = body.str.replace(WHITESPACE_RE, " ", regex=True).str.strip()

    df["content_clean"] = body
    df["content_lower"] = body.str.lower()
    df["has_emoji"] = df["emoji"].str.len() > 0

    log(f"Emoji present in {df['has_emoji'].mean():.1%} of reviews.")
    log(f"Empty after cleaning: {(df['content_clean'] == '').sum():,}")
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """§3.1 — Date parsing and derived temporal features."""
    section("§3.1  Date parsing")

    df["at"] = pd.to_datetime(df["at"], format="mixed", errors="coerce")
    df["repliedAt"] = pd.to_datetime(df["repliedAt"], format="mixed", errors="coerce")

    df["year"] = df["at"].dt.year.astype("Int16")
    df["quarter"] = df["at"].dt.quarter.astype("Int8")
    df["month"] = df["at"].dt.month.astype("Int8")
    df["year_month"] = df["at"].dt.to_period("M").astype(str)

    df["days_to_reply"] = (df["repliedAt"] - df["at"]).dt.total_seconds() / 86400
    df.loc[df["days_to_reply"] < 0, "days_to_reply"] = np.nan

    released = pd.to_datetime(df["released_dt"], errors="coerce")
    df["app_age_at_review"] = (df["at"] - released).dt.days

    span = df["at"].dropna()
    if len(span):
        log(f"Temporal coverage: {span.min():%Y-%m} → {span.max():%Y-%m}")
    log(f"Unparseable dates: {df['at'].isna().sum():,}")
    return df


def compute_review_features(df: pd.DataFrame) -> pd.DataFrame:
    """§3.1 — Derived review-level features."""
    section("§3.1  Review-level features")

    df["n_chars"] = df["content_clean"].str.len().astype("int32")
    df["n_words"] = (
        df["content_clean"].str.split().map(len).astype("int32")
    )
    df["has_reply"] = df["repliedAt"].notna()

    log(f"Median length: {df['n_words'].median():.0f} words")
    log(f"Under {MIN_WORDS_FOR_TEXT_CORPUS} words: {(df['n_words'] < MIN_WORDS_FOR_TEXT_CORPUS).mean():.1%}")
    log(f"Developer reply rate: {df['has_reply'].mean():.1%}")
    return df


# ─── §3.2 Language ──────────────────────────────────────────────────────────────

def _ensure_fasttext_model() -> Path:
    """Download lid.176.bin on first use (~126 MB)."""
    if FASTTEXT_PATH.exists():
        return FASTTEXT_PATH
    log(f"Downloading fastText lid.176 → {FASTTEXT_PATH} (~126 MB, one time)")
    FASTTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(FASTTEXT_URL, FASTTEXT_PATH)  # noqa: S310
    return FASTTEXT_PATH


def detect_language(df: pd.DataFrame) -> pd.DataFrame:
    """
    §3.2 — Language detection

    fastText lid.176 rather than langdetect: half this corpus is under five
    words, where langdetect is unreliable. Falls back to a script-based
    heuristic if fasttext is unavailable so the pipeline still completes.
    """
    section("§3.2  Language detection")

    texts = df["content_clean"].fillna("").str.replace("\n", " ").tolist()

    try:
        import fasttext
        fasttext.FastText.eprint = lambda *a, **k: None  # silence the load banner
        model = fasttext.load_model(str(_ensure_fasttext_model()))
        labels, probs = model.predict(texts, k=1)
        df["lang"] = [lab[0].replace("__label__", "") if lab else "unk" for lab in labels]
        df["lang_conf"] = [float(p[0]) if len(p) else 0.0 for p in probs]
    except Exception as exc:  # noqa: BLE001
        log(f"fastText unavailable ({exc}); using Unicode-script heuristic.", level="WARN")
        df["lang"] = [_script_guess(t) for t in texts]
        df["lang_conf"] = 0.5

    # Empty strings get no meaningful label.
    df.loc[df["content_clean"].str.strip() == "", ["lang", "lang_conf"]] = ["unk", 0.0]

    dist = df["lang"].value_counts().head(10)
    log("Top languages:")
    for lang, n in dist.items():
        log(f"    {lang:>5s}  {n:>8,}  ({n / len(df):.1%})")

    xtab = pd.crosstab(df["app_name"], df["lang"], normalize="index")
    top_langs = [c for c in dist.index[:5] if c in xtab.columns]
    log("Language mix per app (top 5 languages, share of that app's reviews):")
    for app, row in xtab[top_langs].iterrows():
        log(f"    {str(app)[:34]:34s} " + "  ".join(f"{c}={row[c]:.0%}" for c in top_langs))

    return df


def _script_guess(text: str) -> str:
    """Crude fallback: classify by dominant Unicode block."""
    if not text.strip():
        return "unk"
    counts = {"bn": 0, "ar": 0, "en": 0}
    for ch in text:
        cp = ord(ch)
        if 0x0980 <= cp <= 0x09FF:
            counts["bn"] += 1
        elif 0x0600 <= cp <= 0x06FF:
            counts["ar"] += 1
        elif ch.isascii() and ch.isalpha():
            counts["en"] += 1
    if not any(counts.values()):
        return "unk"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def translate_non_english(df: pd.DataFrame, sample_size: int = 0) -> pd.DataFrame:
    """
    §3.2 — Translation

    DESIGN DECISION (deviates from the plan, deliberately): sentiment for
    non-English reviews is scored natively by a multilingual model in 02b
    rather than by translating first. Translating 100K+ reviews through
    deep-translator is neither affordable nor reliable, and the plan itself
    (§16.3) warns that Bengali sentiment does not survive machine translation.

    Translation is therefore reserved for the small samples a human has to
    read: gold-set rows and the verbatim quotes exported in Phase 6.
    Pass --translate-sample N to populate `content_en` for N reviews.
    """
    section("§3.2  Translation (sampled, for human-readable output only)")

    df["content_en"] = pd.NA
    df["translated"] = False

    eligible = df[(df["n_words"] >= MIN_WORDS_FOR_TEXT_CORPUS) & (df["lang"] != "en")]
    log(f"Non-English reviews in the text corpus: {len(eligible):,}")

    if sample_size <= 0:
        log("--translate-sample not set → skipping translation "
            "(02b scores these natively with twitter-xlm-roberta).")
        return df

    try:
        from deep_translator import GoogleTranslator
        from tqdm.auto import tqdm
    except ImportError:
        log("deep-translator not installed; skipping translation.", level="WARN")
        return df

    take = eligible.sample(min(sample_size, len(eligible)), random_state=0)
    log(f"Translating {len(take):,} sampled reviews (throttled; expect a few minutes)...")

    out = {}
    for idx, row in tqdm(take.iterrows(), total=len(take), desc="translate", file=sys.stderr):
        try:
            out[idx] = GoogleTranslator(source="auto", target="en").translate(
                row["content_clean"][:4500]
            )
        except Exception:  # noqa: BLE001 — network/rate errors are expected; skip the row
            continue

    df.loc[list(out), "content_en"] = pd.Series(out)
    df.loc[list(out), "translated"] = True
    log(f"Translated {len(out):,} reviews.")
    return df


# ─── §3.3 Sub-corpora ───────────────────────────────────────────────────────────

def assign_subcorpora(df: pd.DataFrame) -> pd.DataFrame:
    """
    §3.3 — Analysis sub-corpora as boolean flags on one frame.

    One source of truth beats four physical files: joins stay simple and a
    row can belong to several corpora without being duplicated.
    """
    section("§3.3  Analysis sub-corpora")

    feat = load_feature_csv()
    annotated_keys = set(feat.loc[feat["is_annotated"], "app_key"])

    df["corpus_all"] = True
    df["corpus_text"] = (df["n_words"] >= MIN_WORDS_FOR_TEXT_CORPUS) & (df["content_clean"].str.strip() != "")
    df["corpus_en"] = df["corpus_text"] & (df["lang"] == "en")
    df["corpus_trans"] = df["corpus_text"] & (df["lang"] != "en")
    df["corpus_annot"] = df["corpus_text"] & df["app_key"].isin(annotated_keys)

    matched = df.loc[df["app_key"].isin(annotated_keys), "app_name"].nunique()
    log(f"Apps matched to the annotation sheet: {matched}")
    unmatched = sorted(set(df.loc[~df["app_key"].isin(annotated_keys), "app_name"].dropna().unique()))
    if unmatched:
        log(f"Apps NOT in the annotated set ({len(unmatched)}): {', '.join(str(a)[:30] for a in unmatched)}")

    log("")
    log("Sub-corpus sizes:")
    for c in ["corpus_all", "corpus_text", "corpus_en", "corpus_trans", "corpus_annot"]:
        n = int(df[c].sum())
        log(f"    {c:14s} {n:>8,}  ({n / len(df):.1%})")
    return df


# ─── Pipeline ───────────────────────────────────────────────────────────────────

def main() -> None:
    p = base_parser(__doc__ or "Phase 0 — preprocessing")
    p.add_argument(
        "--translate-sample", type=int, default=0, metavar="N",
        help="Translate N non-English reviews into content_en for human reading. "
             "0 disables translation (sentiment is scored natively in 02b).",
    )
    args = p.parse_args()
    set_seed(args.seed)

    df = load_and_merge_reviews()
    df = maybe_sample(df, args)
    df = normalize_text(df)
    df = parse_dates(df)
    df = compute_review_features(df)
    df = detect_language(df)
    df = translate_non_english(df, args.translate_sample)
    df = assign_subcorpora(df)

    section("Writing output")
    drop = [c for c in ("userImage",) if c in df.columns]
    df = df.drop(columns=drop)

    MASTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MASTER_PATH, index=False, compression="zstd")
    summarize(df, "master_reviews")
    log(f"Saved → {MASTER_PATH}")

    log("")
    log("Next: python src/scripts/02a_sentiment_vader.py")


if __name__ == "__main__":
    main()
