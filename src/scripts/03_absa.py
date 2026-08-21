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
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "master_reviews_with_sentiment.parquet"
LEXICON_PATH = PROJECT_ROOT / "data" / "aspect_lexicon.yaml"
OUTPUT_PATH = PROJECT_ROOT / "data" / "aspect_sentiments.parquet"

# ─── 20 CSV-linked aspects (§5.1) ──────────────────────────────────────────────
CSV_LINKED_ASPECTS = [
    "prayer_times_accuracy",
    "madhab",
    "calc_method",
    "reminders_adhan",
    "forbidden_times",
    "nafl_times",
    "goal_system",
    "mosque_finder",
    "guides",
    "adhkar",
    "qibla",
    "table_format",
    "qasr_travel",
    "calendar_sync",
    "monetization",
    "prayer_tracker",
    "tracker_score",
    "women_period",
    "widgets",
    "companion_hardware",
]

# ─── 7 Review-only aspects (§5.2) ──────────────────────────────────────────────
REVIEW_ONLY_ASPECTS = [
    "ui_design",
    "stability_bugs",
    "ads_intrusive",
    "privacy_data",
    "quran_audio",
    "complexity_bloat",
    "spiritual_affect",
]

ALL_ASPECTS = CSV_LINKED_ASPECTS + REVIEW_ONLY_ASPECTS


def load_aspect_lexicon() -> dict:
    """
    Load aspect taxonomy from data/aspect_lexicon.yaml

    TODO:
    - Parse YAML file with seed keywords, regex patterns per aspect
    - Include transliterations: azan/athan/adhan, zikr/dhikr, mazhab/madhab, qaza/qada
    - Include Bengali/Arabic surface forms for untranslated path
    - Return dict: {aspect_name: [keywords/patterns]}
    """
    pass


def segment_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """
    §5.3 Step 1 — Sentence segmentation

    TODO:
    - Use pysbd (robust to informal text) to split each review into sentences
    - Return expanded DataFrame with one row per (reviewId, sentence_index, sentence_text)
    """
    pass


def lexicon_tagging(sentences_df: pd.DataFrame, lexicon: dict) -> pd.DataFrame:
    """
    §5.3 Step 2 — Lexicon tagging

    TODO:
    - For each sentence, check against expanded keyword/regex sets per aspect
    - A sentence may carry MULTIPLE aspects
    - Return DataFrame with columns: reviewId, sentence_index, aspect, match_method='lexicon'
    """
    pass


def zero_shot_backstop(sentences_df: pd.DataFrame, tagged_df: pd.DataFrame) -> pd.DataFrame:
    """
    §5.3 Step 3 — Zero-shot classification for unmatched sentences

    TODO:
    - Use facebook/bart-large-mnli
    - Run ONLY on sentences with NO lexicon hit in C_text (keeps cost bounded)
    - Accept above a tuned threshold (calibrate on 300-review aspect gold set)
    - Return additional aspect assignments with match_method='zero-shot'
    """
    pass


def per_aspect_sentiment(tagged_df: pd.DataFrame) -> pd.DataFrame:
    """
    §5.3 Step 4 — Per-aspect sentiment

    TODO:
    - Run RoBERTa on the triggering sentence(s) for each aspect hit
    - Aggregate to review×aspect by probability-weighted mean
    - Store: sentiment_label, prob_neg, prob_neu, prob_pos
    """
    pass


def handle_requests_and_negation(tagged_df: pd.DataFrame) -> pd.DataFrame:
    """
    §5.3 Step 5 — Negation and request handling

    TODO:
    - Detect request patterns: "wish it had", "please add", "should have", etc.
    - A sentence matching a request pattern must NOT be scored as negative sentiment
      toward an existing feature — route to Phase 2.5 demand mining instead
    - Set is_request = True for these
    - This is a common and paper-killing failure mode in keyword ABSA
    """
    pass


def main():
    """
    ABSA pipeline:
    1. Load lexicon
    2. Segment sentences
    3. Lexicon tagging
    4. Zero-shot backstop (unmatched sentences only)
    5. Handle requests and negation
    6. Compute per-aspect sentiment
    7. Save to aspect_sentiments.parquet
    """
    # TODO: Wire up pipeline
    pass


if __name__ == "__main__":
    main()
