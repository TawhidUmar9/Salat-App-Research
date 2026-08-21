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
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "master_reviews_with_sentiment.parquet"
ASPECT_PATH = PROJECT_ROOT / "data" / "aspect_sentiments.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "demand.parquet"

# ─── Pattern categories (§6.2) ──────────────────────────────────────────────────

# Request patterns — explicit feature requests
REQUEST_PATTERNS = [
    # TODO: Compile these as regex patterns
    # wish, hope, please add, should have, would be (nice|better|great) if,
    # needs? (a|an|to), missing, no option, can you add, kindly add,
    # suggestion, lacks
]

# Absence patterns — stating a feature doesn't exist
ABSENCE_PATTERNS = [
    # TODO:
    # doesn't have, does not support, no <aspect>, couldn't find
]

# Switching / churn patterns — feeds RQ4 loyalty and RQ3 competitive edge
CHURN_PATTERNS = [
    # TODO:
    # uninstall, deleted, switched to, moved to, better than, going back to
]

# Preference-citation patterns — operationalization of "competitive edge"
PREFERENCE_PATTERNS = [
    # TODO:
    # that's why I use, only app that, no other app, the reason I
]


def extract_request_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO:
    - Apply REQUEST_PATTERNS regex to review sentences
    - Attach each match to an aspect (from aspect taxonomy)
    - Set request_type = 'request'
    - Return matches with evidence_sentence
    """
    pass


def extract_absence_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO:
    - Apply ABSENCE_PATTERNS regex
    - Attach to aspect
    - Set request_type = 'absence'
    """
    pass


def extract_churn_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO:
    - Apply CHURN_PATTERNS regex
    - Attach to aspect where possible
    - Set request_type = 'churn'
    - These feed RQ4 loyalty and RQ3 competitive edge analyses
    """
    pass


def extract_preference_citations(df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO:
    - Apply PREFERENCE_PATTERNS regex
    - Attach to aspect
    - Set request_type = 'preference'
    - This is the operationalization of "competitive edge" for RQ3
    """
    pass


def precision_check_sample(demand_df: pd.DataFrame, sample_size: int = 200) -> pd.DataFrame:
    """
    TODO:
    - Sample 200 demand matches for manual precision check
    - Save to data/gold_labels/demand_precision_sample.csv
    - Report precision (these patterns are noisy)
    """
    pass


def main():
    """
    Demand mining pipeline:
    1. Load reviews and aspect sentiments
    2. Extract request patterns
    3. Extract absence patterns
    4. Extract churn signals
    5. Extract preference citations
    6. Combine all, attach to aspects
    7. Generate precision-check sample
    8. Save to demand.parquet
    """
    # TODO: Wire up pipeline
    pass


if __name__ == "__main__":
    main()
