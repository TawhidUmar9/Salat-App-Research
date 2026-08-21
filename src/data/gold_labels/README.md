# Gold Labels Directory
# =====================
# This directory holds annotation files for validation.
#
# Expected files:
#   doc_500.csv     — 500 document-level reviews, stratified by:
#                     install tier (>100M / 10M–100M / 1M–10M / <1M)
#                     × star (1–2 / 3 / 4–5) × language (en / translated)
#                     Labels: {Positive, Negative, Neutral, Mixed}
#
#   aspect_300.csv  — 300 reviews labelled for aspect + per-aspect sentiment
#                     (Phase 2 validation)
#
#   demand_precision_sample.csv — 200 demand-mining matches for precision check
#
# See implementation_plan.md §4.4, §12 for specification.
