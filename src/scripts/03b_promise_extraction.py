"""
Phase 2.5a — Promise Extraction
================================
Extract claimed features from Play Store descriptions for all 26 apps.
This covers the 6 unannotated apps and provides a second measurement
for inter-source agreement (Cohen's κ) against the 20 hand-annotated apps.

Input:
    - reviews/*/metadata.json          (26 app descriptions)
    - Salah App Analysis - Sheet1.csv  (20 hand-annotated apps)
    - data/aspect_lexicon.yaml         (shared taxonomy)

Output:
    - data/feature_matrix_combined.parquet
      Schema: app × feature × {csv_annotated, description_claimed, source_agreement}

See implementation_plan.md §6.1 for full specification.
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEWS_DIR = PROJECT_ROOT.parent / "reviews"
CSV_PATH = PROJECT_ROOT.parent / "Salah App Analysis - Sheet1.csv"
LEXICON_PATH = PROJECT_ROOT / "data" / "aspect_lexicon.yaml"
OUTPUT_PATH = PROJECT_ROOT / "data" / "feature_matrix_combined.parquet"


def load_app_descriptions() -> pd.DataFrame:
    """
    TODO:
    - Load metadata.json from each of the 26 app directories
    - Extract: app_name, package, description text, adSupported, offersIAP, released
    - Return DataFrame with one row per app
    """
    pass


def segment_description(description: str) -> list[str]:
    """
    TODO:
    - Segment each description into feature bullets/sentences
    - Handle bullet-point lists, newlines, and paragraph breaks
    - Return list of text segments
    """
    pass


def classify_features(segments: list[str]) -> dict[str, bool]:
    """
    TODO:
    - Classify each segment against the 22-feature taxonomy
    - Use lexicon matching first, then zero-shot for ambiguous segments
    - Return dict: {feature_name: True/False} with supporting sentence retained as evidence
    - Produces promised[app, feature] ∈ {0, 1}
    """
    pass


def validate_against_csv(description_matrix: pd.DataFrame, csv_matrix: pd.DataFrame) -> dict:
    """
    TODO:
    - Compare description-derived features against the 20 hand-annotated apps
    - Compute Cohen's κ per feature
    - Generate confusion table
    - Report inter-source agreement — this is a reportable validity contribution
    """
    pass


def check_monetization_claims(df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO:
    - Combine adSupported / offersIAP from metadata with
      'All Features for free' annotation to check monetization claims
    - Flag contradictions (claims "free" but has IAP)
    """
    pass


def main():
    """
    Promise extraction pipeline:
    1. Load all 26 app descriptions from metadata.json
    2. Segment descriptions into feature bullets/sentences
    3. Classify each segment against the 22-feature taxonomy
    4. Load CSV annotations for the 20 annotated apps
    5. Validate description-derived vs. CSV (Cohen's κ)
    6. Check monetization claims
    7. Combine into feature_matrix_combined.parquet
    """
    # TODO: Wire up pipeline
    pass


if __name__ == "__main__":
    main()
