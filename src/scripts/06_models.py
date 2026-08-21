"""
Phase 5 — Statistical Modelling
=================================
Review-level mixed-effects models (M1–M6) for RQ1–RQ6.
All models use a random intercept per app.

Input:
    - data/master_reviews_with_sentiment.parquet
    - data/aspect_sentiments.parquet
    - data/feature_matrix_combined.parquet
    - data/demand.parquet

Output:
    - data/app_comparison.csv
    - data/gap_matrix.csv (RQ6)

See implementation_plan.md §9 for full specification.
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEWS_PATH = PROJECT_ROOT / "data" / "master_reviews_with_sentiment.parquet"
ASPECT_PATH = PROJECT_ROOT / "data" / "aspect_sentiments.parquet"
FEATURE_PATH = PROJECT_ROOT / "data" / "feature_matrix_combined.parquet"
DEMAND_PATH = PROJECT_ROOT / "data" / "demand.parquet"
OUTPUT_COMPARISON = PROJECT_ROOT / "data" / "app_comparison.csv"
OUTPUT_GAP = PROJECT_ROOT / "data" / "gap_matrix.csv"


def model_m1_rq2_competing_predictors(df: pd.DataFrame):
    """
    §9.1 — M1: RQ2 core — competing predictors of dissatisfaction

    TODO:
    - Formula:
        score_ij ~ complaint_accuracy + complaint_qibla + complaint_madhab
                 + complaint_calcmethod + complaint_ui + complaint_bugs
                 + complaint_ads + complaint_reminders
                 + log(n_words) + review_year + (1 | app)
    - Fit as linear mixed model (statsmodels MixedLM)
    - Robustness check: ordinal mixed model (star is ordinal, ceiling-heavy)
    - The paper's RQ2 answer = comparison of accuracy-family coefficients
      against complaint_ui coefficient
    - Report standardized effects with CIs, not just p-values
    """
    pass


def model_m2_rq1_gamification_valence(df: pd.DataFrame, aspect_df: pd.DataFrame):
    """
    §9.2 — M2: RQ1 — gamification valence

    TODO:
    - Filter to prayer_tracker ∨ tracker_score ∨ goal_system reviews
    - sentiment ~ tracker_tier (none / tracker only / tracker + score) + (1 | app)
    - Affect split:
        - Rate of spiritual_affect-negative (guilt, pressure, anxiety, shame)
        - vs. spiritual_affect-positive (motivation, discipline, consistency)
        - WITHIN tracker reviews vs. corpus baseline
    - The love/hate tension is the RATIO, not the mean sentiment
    - A mean can hide a bimodal split — report distribution shape
    - Run bimodality/dip test (Hartigan's dip test)
    - Qualitative coding of 50 tracker-negative reviews for thematic layer
    """
    pass


def model_m3_rq4_inclusion(df: pd.DataFrame, aspect_df: pd.DataFrame):
    """
    §9.3 — M3: RQ4 — bio-spiritual inclusion

    TODO:
    PRIMARY (well-powered):
    - Volume and sentiment of women_period reviews across all 26 apps
    - Demand rate for menstrual exemption among apps lacking it
    - Qualitative coding of the full set (few hundred to few thousand reviews)
    - If n is small, that scarcity is itself the finding

    SECONDARY (underpowered, label as such):
    - 3 apps with Women tracking vs. 17 without
    - Descriptive comparison with bootstrap CIs
    - Explicit statement that N=3 precludes causal inference
    - Note confound: Islamic Habit Tracker is one of three, has lowest CSV rating (3.8)

    LOYALTY PROXIES (label as proxies):
    - Tenure mentions: regex 'using (this|it) for \\d+ (years|months)', 'since 20\\d\\d'
    - Churn signals: uninstall, deleted, switched to, going back to
    - Endorsement: thumbsUpCount distribution
    - Rating trajectory: per-app rating slope over time
    """
    pass


def model_m4_rq5_bloat_hardware(df: pd.DataFrame, feature_df: pd.DataFrame):
    """
    §9.4 — M4: RQ5 — feature bloat vs. companion hardware

    TODO:
    - Compute feature_count (0–22) per app from feature_matrix_combined
      (description-derived for the 6 unannotated)

    REVIEW-LEVEL:
    - P(complexity_bloat complaint) and P(stability_bugs complaint)
      ~ feature_count + log(realInstalls) + adSupported + (1 | app)
    - Logistic mixed model

    APP-LEVEL (n=20/26, exploratory only):
    - Spearman ρ of feature_count vs. mean sentiment and vs. complaint rates
    - Bootstrap CIs, explicitly framed as exploratory

    HARDWARE CASE STUDY:
    - iQIBLA Life companion_hardware reviews (n≈? from 4,294 total)
    - Demand rate for wearable/watch support across the other 25 apps

    CONFOUND TO NAME: feature count correlates with install base and company size.
    log(realInstalls) is a required control; residual confounding must be acknowledged.
    """
    pass


def model_m5_rq3_competitive_edge(df: pd.DataFrame, aspect_df: pd.DataFrame, demand_df: pd.DataFrame):
    """
    §9.5 — M5: RQ3 — competitive edge

    TODO:
    MOSQUE FINDER (8/20, testable):
    - Preference-citation rate and mosque_finder sentiment
    - With (1 | app)

    AUTO QASR (1/20, NOT testable as between-app comparison):
    - Report demand volume across the 19 apps lacking it
    - Qasr/travel complaint profile inside those apps
      (mistimed prayers after travel, timezone bugs)
    - The PROBLEM is measurable ecosystem-wide even though
      the SOLUTION exists in only one app
    """
    pass


def model_m6_rq6_gap_matrix(feature_df: pd.DataFrame, aspect_df: pd.DataFrame, demand_df: pd.DataFrame):
    """
    §9.6 — M6: RQ6 — the gap matrix

    TODO:
    6a DELIVERY GAP:
    - For each (app, feature) where promised = 1:
      compute negative-sentiment rate on corresponding aspect + mention volume
    - Flag neg_rate > 40% AND n >= 30 as a "broken promise"
    - Output app × feature matrix

    6b COVERAGE GAP:
    - For each (app, feature) where promised = 0:
      compute demand rate from demand.parquet
    - Aggregate to ecosystem level: which needs does NO app serve well?
    - This directly answers "are current apps enough"
    - Rank unmet needs by (demand volume × number of apps failing to serve them)
    - This ranked list bridges to the Falah design-implications section

    Save to gap_matrix.csv
    """
    pass


def multiple_comparisons_correction(results: pd.DataFrame) -> pd.DataFrame:
    """
    §9.7 — Multiple comparisons & reporting

    TODO:
    - Apply Benjamini–Hochberg FDR correction (NOT Bonferroni —
      too conservative with 22 features × multiple aspects, guarantees null)
    - Report both q-values and effect sizes
    - Make effect sizes with CIs the headline
    - Every rare-feature test carries a stated power caveat in table caption
    """
    pass


def developer_response_analysis(df: pd.DataFrame):
    """
    §9.8 — Developer response analysis

    TODO:
    - Reply rate per app (verified range 1.1%–78.8%)
    - Sentiment of replies
    - days_to_reply distribution
    - Whether reply presence correlates with later rating trajectory
    - NOTE: reply-vs-rating is CORRELATIONAL and reply targeting is non-random
      (devs reply to negative reviews selectively) — do NOT claim causation
    """
    pass


def main():
    """
    Statistical modelling pipeline:
    1. Load all input data
    2. Run M1 (RQ2): competing predictors
    3. Run M2 (RQ1): gamification valence
    4. Run M3 (RQ4): inclusion
    5. Run M4 (RQ5): bloat/hardware
    6. Run M5 (RQ3): competitive edge
    7. Run M6 (RQ6): gap matrix
    8. Apply multiple comparisons correction
    9. Developer response analysis
    10. Save app_comparison.csv and gap_matrix.csv
    """
    # TODO: Wire up pipeline
    pass


if __name__ == "__main__":
    main()
