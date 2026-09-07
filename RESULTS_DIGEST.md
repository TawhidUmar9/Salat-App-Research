# Results digest — every reportable figure, one page

Companion to [`PAPER_DISCLOSURES.md`](PAPER_DISCLOSURES.md). That file says what
must be disclosed; this one is the lookup table so nobody re-derives a number
from terminal scrollback at 2am.

**Source of truth:** `src/data/model_results.csv` (2026-09-07 run) for every
coefficient; the four `topic_info*.csv` sheets and two coding sheets for
qualitative counts. ✅ = verified in-repo on 2026-09-07. 📋 = from the run, not
re-derived here.

---

## Corpus

| Measure | Value | |
|---|---|---|
| Reviews scraped | 732,194 across 26 apps | 📋 |
| Analysed (has text) | 324,687 | 📋 |
| Topic-modelled (C_en) | 276,422 | ✅ |
| Krippendorff's α | 0.824 | 📋 |
| Aspect precision, weighted | 84.4% [79.7, 88.9] on 192/300 gold rows | 📋 |
| Aspect precision, unweighted | 69.3% | 📋 |
| Demand-pattern precision | 91.5% (`absence` 65%) | 📋 |
| Promise-source agreement | mean κ = 0.272 | 📋 |
| BH-FDR survivors | 20 of 29 at q < 0.05 | ✅ |
| Cannot support inference | 11 (3 no SE, 8 non-converged) | ✅ |

---

## RQ1 — Gamification

| Quantity | Value | Caveat |
|---|---|---|
| Tier effect, tracker-only | β = −0.087 [−0.332, 0.158], p = 0.488 | **cluster-robust OLS**, 25 clusters |
| Tier effect, tracker+score | β = −0.134 [−0.415, 0.147], p = 0.349 | same |
| M2 mixed model | **do not cite** | SE 2.0e6 / 1.3e6, p ≈ 0.99999 |
| Bimodality | Hartigan's D = 0.118, n = 6,285 | write **p < 1e-16**, never p = 0 |
| Guilt vs motivation | Fisher OR = 0.283, p = 3.5e-11, q = 6.7e-11, n = 149 | ratio 0.26 vs baseline 0.86 📋 |
| Guilt in coded reviews | 3 of 77 (3.9%) | both passes, different apps |

**Theme prevalence** (never pool the two passes):

| Theme | Pass 1 (Athan, n=49) | Pass 2 (22 apps, n=28) |
|---|---|---|
| C. Record can't be trusted | 11 (22%) | 6 (21%) |
| D. Normative mismatch | 6 (12%) | 3 (11%) |
| E. Prompting failures | 7 (14%) | 13 (46%) |
| A. Redesign as regression | 10 (20%) | 0 |
| B. Retroactive logging denied | 5 (10%) | 0 |
| F. Monetization vs religious purpose | 10 (20%) | 1 (4%) |
| G. Record gated, not owned | — | 3 (11%) |
| H. App overreaches into device | — | 2 (7%) |

---

## RQ2 — Accuracy vs interface

**M1 review-level mixed model, n = 324,687.** All eight survive q < 0.05.

| Complaint | β | 95% CI |
|---|---|---|
| Intrusive ads | −1.439 | [−1.460, −1.418] |
| Prayer-time accuracy | −1.112 | [−1.142, −1.083] |
| Stability / bugs | −0.938 | [−0.965, −0.910] |
| Reminders / adhan | −0.845 | [−0.868, −0.823] |
| UI design | −0.600 | [−0.664, −0.537] |
| Calculation method | −0.600 | [−0.769, −0.430] |
| Qibla | −0.550 | [−0.594, −0.507] |
| Madhab | −0.324 | [−0.526, −0.122] |

- **Accuracy-family mean −0.647** vs interface −0.600 — close as classes ✅
- **Prayer-time accuracy alone −1.112**, interval disjoint from interface, 1.85× ✅
- `madhab` ordinal check: β = +0.216, p = 0.649 — **sign flip**, keep the caveat ✅
- **Star masking:** of 7,942 reviews with a negative accuracy complaint, **3,306
  (41.6%) gave 4–5 stars; 1,830 (23.0%) gave 5** ✅
  (1★ 2,319 · 2★ 914 · 3★ 1,403 · 4★ 1,476 · 5★ 1,830)
- Qibla failure mode: **compass instability**, not miscalculation ✅

---

## RQ3 — Travel / competitive edge

| Quantity | Value | |
|---|---|---|
| Mosque finder, cluster-robust | β = −0.137, p = 0.003, 21 clusters, n = 3,010 | ✅ |
| Mosque finder, mixed model | +0.05, **NaN SE — cannot arbitrate** | ✅ |
| Mentions, apps *with* feature | 1,421 / 10 apps — 27.0% neg, 38.6% pos | ✅ |
| Mentions, apps *without* | 1,589 / 11 apps — 23.0% neg, 43.6% pos | ✅ |
| Mosque-finder demand where absent | 243 signals, 9 apps | ✅ |
| Preference citations, total | 4,696 | ✅ |
| Aspects cited | adhan 374 · Quran audio 293 · accuracy 290 · qibla 155 · adhkar 152 | ✅ |
| Citation rate | 1.38% with mosque finder vs 1.24% without | ✅ |
| Qasr: apps implementing | **1 of 26** (Muslim Bangla) | ✅ |
| Qasr demand | 21 signals, 20 from 5 apps lacking it | ✅ |

⚠️ `qasr_travel` shows 636 mentions at 76.7% positive — **the lexicon
over-captures travel/pilgrimage talk. Never report 636 as qasr volume.** ✅

---

## RQ4 — Inclusion

| Quantity | Value | |
|---|---|---|
| Mentions | 2,194 across 24 of 26 apps | ✅ |
| Distinct reviews | 2,182 (0.672% of corpus) | ✅ |
| Apps with the feature | **3** (Athan, Islamic Habit Tracker, Pillars) | ✅ |
| Sentiment, apps *with* | 26.3% negative (464 mentions, 3 apps) | ✅ |
| Sentiment, apps *without* | 26.0% negative (1,718 mentions, 21 apps) | ✅ |
| Demand | 272 total; 216 from 16 apps lacking it | ✅ |
| Demand types | request 161 · preference 51 · churn 36 · absence 21 · tenure 3 | ✅ |
| Δ star | −0.236 (4.135 vs 4.371) — **NOT ESTIMABLE** | ✅ |

Menstrual complaints appear in **both** coding passes, different apps ✅

---

## RQ5 — Bloat and hardware

| Quantity | Value | |
|---|---|---|
| feature_count → bloat | β = −0.000487, p = 0.272 (**cluster-robust**) | ✅ |
| M4 `feature_count` | **do not cite** — SE 743.8, p ≈ 0.99999 | ✅ |
| Co-occurrence, all | lift **23.98×**, OR 29.41, p = 7.4e-137 | ✅ |
| Co-occurrence, separate sentences | lift **7.62×**, OR 8.78, p = 9.3e-24 | ✅ |
| Fused reviews removed | 94 of 134 (70% fuse, 30% separate) | ✅ |
| 5 leanest apps (6–9 features) | 3.96 stars, 5.2% bloat rate | ✅ |
| 5 fullest apps (13–14) | 4.71 stars, 1.6% bloat rate | ✅ |
| Robustness (drop n=5, n=45) | 4.20 stars, 5.6% | ✅ |
| Cleanest pair | iPray 8 feat/8.0% vs Athan 13 feat/1.5% | ✅ |
| features ~ installs | Spearman ρ = 0.244, n = 26 | ✅ |
| Hardware, inside iQIBLA | 37.7% neg, 45.9% pos (n = 518) | ✅ |
| Hardware, other 23 apps | 49.2% neg, 20.8% pos (n = 1,994) | ✅ |
| Hardware mentions total | 2,482 across 24 apps | ✅ |

Corpus mean star 4.40 ✅ · `rq5_bloat` ARI 0.452, 6–11 topics 📋

---

## RQ6 — Gaps

| Quantity | Value | |
|---|---|---|
| Promised pairs | 272; 136 judgeable at n ≥ 30 | ✅ |
| Broken promises | **18 across 11 apps** | ✅ |
| Most broken by share | monetization 5/16 (31%) · adhan 5/20 (25%) | ✅ |
| Largest by volume | Muslim Pro adhan 9,989 @ 45.6% · monetization 9,182 @ 54.8% | ✅ |
| Served well | madhab, monetization, qibla, table_format, widgets | ✅ |
| Coverage < 25% | calendar_sync, companion_hardware, qasr_travel, women_period | ✅ |

**Unmet-need ranking** (demand × apps lacking):

| Aspect | Score | Apps lacking |
|---|---|---|
| companion_hardware | 6,725 | 25 |
| women_period | 4,738 | 23 |
| mosque_finder | 3,304 | 14 |
| forbidden_times | 2,805 | 15 |
| nafl_times | 2,445 | 15 |
| prayer_tracker | 1,905 | 15 |

⚠️ `tracker_score` 1/1 broken — **one promising app, ratio meaningless** ✅

---

## Figures that must never appear

| Wrong | Right |
|---|---|
| "record integrity 52%" | C at ~21%; A and B are app-specific |
| "no tier effect (M2)" | cite the cluster-robust OLS |
| "the accuracy coefficient is −1.112" | that is prayer-time accuracy, one of four |
| "p = 0" for the dip test | p < 1e-16 |
| "636 qasr mentions" | 21 demand signals |
| "3 models not estimable" | 11 cannot support inference |
| "tracker_score most broken" | 1 of 1 app; report adhan and monetization |
| pooled coding percentages | two passes, reported separately |
