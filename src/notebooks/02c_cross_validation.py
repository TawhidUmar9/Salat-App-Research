# Phase 1C — Cross-Validation Notebook
# ======================================
# Validate sentiment methods against each other and against gold labels.
#
# See implementation_plan.md §4.4 for full specification.
#
# This file is a placeholder. Convert to Jupyter notebook (.ipynb) before use.
#
# ─── Contents ────────────────────────────────────────────────────────────────────
#
# 1. Load sentiment results from all three methods:
#    - VADER (02a)
#    - RoBERTa (02b)
#    - Star-rating proxy (02b)
#
# 2. Cross-method validation:
#    - Cohen's κ for all three pairs (VADER↔RoBERTa, VADER↔Star, RoBERTa↔Star)
#    - Confusion matrices for each pair
#
# 3. Gold set evaluation (500 document-level reviews):
#    - Stratified by:
#        - Install tier: >100M / 10M–100M / 1M–10M / <1M
#        - Star: 1–2 / 3 / 4–5
#        - Language: en / translated
#    - Labels: {Positive, Negative, Neutral, Mixed}
#    - Precision / recall / F1 per method
#    - Krippendorff's α if ≥2 annotators
#      (recommended: 2 annotators on 100-review overlap, then split remainder)
#
# 4. Aspect-level gold set (300 reviews):
#    - Labelled for aspect + per-aspect sentiment
#    - Phase 2 is the paper's analytical core — cannot validate with doc-level gold set alone
#
# 5. Generate gold-set labelling spreadsheets:
#    - Stratified sample with pre-filled columns and instructions tab
#    - Save to data/gold_labels/
