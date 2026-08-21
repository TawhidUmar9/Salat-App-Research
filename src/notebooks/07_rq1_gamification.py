# Phase 6 — RQ1 Deep Dive: The Gamification Tension
# ====================================================
# Do users love or hate prayer streaks and grading scores?
#
# See implementation_plan.md §10, §9.2 for full specification.
#
# This file is a placeholder. Convert to Jupyter notebook (.ipynb) before use.
#
# ─── Contents ────────────────────────────────────────────────────────────────────
#
# 1. M2 model results: sentiment ~ tracker_tier + (1 | app)
#
# 2. Tracker sub-topics from BERTopic
#    - streak-anxiety cluster
#    - streak-motivation cluster
#
# 3. Guilt/motivation ratio analysis
#    - spiritual_affect-negative rate: guilt, pressure, anxiety, shame
#    - spiritual_affect-positive rate: motivation, discipline, consistency
#    - Within tracker reviews vs. corpus baseline
#
# 4. Bimodality test (Hartigan's dip test)
#    - Is the sentiment distribution bimodal for tracker reviews?
#
# 5. Qualitative coding of 50 tracker-negative reviews
#    - Thematic analysis layer
#
# 6. Export 10–15 verbatim illustrative quotes
#    - Include app + date + star
#    - Save to data/quotes/rq1.csv
#    - ANONYMIZE userName — do not print reviewer names in the paper
