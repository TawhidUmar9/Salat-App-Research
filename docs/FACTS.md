# Every verified number

✅ verified in-repo 2026-09-09 · 📋 from the run, not re-derived
Analysis run **2026-09-03**, commit **`5da2db7`**, reproducibility confirmed by
re-running `06_models.py` from a clean tree (byte-identical output).

## Corpus
| | | |
|---|---|---|
| Reviews scraped | 732,194 across 26 apps | 📋 |
| Analysed (has text) | 324,687 | 📋 |
| Topic-modelled (C_en) | 276,422 | ✅ |
| Krippendorff's α | 0.824 (100 overlap rows, 89.0% raw agreement) | ✅ |
| BH-FDR family | 29 coefficients, **20 survive q < 0.05** | ✅ |
| Cannot support inference | 11 (3 no SE + 8 non-converged) | ✅ |

## Measurement
| | | |
|---|---|---|
| Aspect precision, unweighted | 69.3% (n = 192 of 300 gold rows) | ✅ |
| Aspect precision, corpus-weighted | 84.4% [79.7, 88.9] | ✅ |
| Demand-pattern precision | 91.5% (183/200) | ✅ |
| — by type | preference 100%, tenure 100%, churn 97.5%, request 95%, **absence 65%** | ✅ |
| Promise-source agreement | mean κ = 0.272 (range 0.62 → 0.36) | ✅ |
| Zero-shot backstop (dropped) | 11.5% precision on 300 sentences | 📋 |
| Devotional formulae filtered | 1,686 sentences (0.8%); `goal_system` 0% precision | 📋 |
| No gold rows | `calendar_sync`, `goal_system` (0.2% of volume) | ✅ |

### ⚠️ Per-aspect precision — seven aspects under 50%
| Aspect | corpus n | precision | 95% CI | use |
|---|---|---|---|---|
| reminders_adhan | 30,069 | 100% | [65,100] | safe |
| quran_audio | 24,964 | 100% | [61,100] | safe |
| prayer_times_accuracy | 21,941 | 73% | [43,90] | safe |
| ads_intrusive | 17,845 | 100% | [74,100] | safe |
| ui_design | 15,720 | 73% | [43,90] | safe |
| monetization | 13,914 | 82% | [52,95] | safe |
| qibla | 8,526 | 82% | [52,95] | safe |
| stability_bugs | 7,665 | 64% | [35,85] | safe |
| privacy_data | 7,471 | 90% | [60,98] | safe |
| prayer_tracker | 6,009 | 78% | [45,94] | safe |
| complexity_bloat | 5,389 | 67% | [35,88] | safe |
| widgets | 4,755 | 100% | [74,100] | safe |
| **mosque_finder** | 3,043 | **27%** | [10,57] | ~821 true |
| **companion_hardware** | 2,512 | **40%** | [12,77] | ~1,004 true |
| **women_period** | 2,194 | **20%** | [6,51] | **~440 true** |
| **forbidden_times** | 1,534 | **10%** | [2,40] | ~153 true |
| **table_format** | 1,483 | **44%** | [19,73] | ~652 true |
| calc_method | 639 | 90% | [60,98] | safe |
| **qasr_travel** | 637 | **0%** | [0,66] | do not report volume |
| madhab | 491 | 60% | [23,88] | thin |
| **tracker_score** | 101 | **0%** | [0,79] | 1 gold row, uninformative |

**Rule:** never report a bare count for the bolded aspects. Model coefficients
are unaffected — all use aspects at 64–100%.

## RQ1 — Gamification
| | | |
|---|---|---|
| Tier, tracker-only | β = −0.087 [−0.332, 0.158], p = 0.488 | **cluster-robust OLS**, 25 clusters |
| Tier, tracker+score | β = −0.134 [−0.415, 0.147], p = 0.349 | same |
| M2 mixed model | **DO NOT CITE** — SE 2.0e6/1.3e6, p ≈ 0.99999 | ✅ |
| Bimodality | Hartigan's D = 0.118, n = 6,285 | write **p < 1e-16** |
| Guilt vs motivation | Fisher OR = 0.283, p = 3.5e-11, q = 6.7e-11, n = 149 | ratio 0.26 vs baseline 0.86 |
| Guilt in coded reviews | **3 of 77 (3.9%)** | both passes |
| Tracker reviews | 6,285 across 25 apps | ✅ |

### Coding passes — REPORT SEPARATELY, NEVER POOLED
| Theme | Pass 1 (Athan, n=49) | Pass 2 (22 apps, n=28) |
|---|---|---|
| C. The record can't be trusted | 11 (22%) | 6 (21%) |
| D. Normative mismatch | 6 (12%) | 3 (11%) |
| E. Prompting failures break the loop | 7 (14%) | **13 (46%)** |
| A. Redesign as regression | 10 (20%) | **0** |
| B. Retroactive logging denied | 5 (10%) | **0** |
| F. Monetization vs religious purpose | 10 (20%) | 1 (4%) |
| G. The record is gated, not owned | — | 3 (11%) *new* |
| H. App overreaches into the device | — | 2 (7%) *new* |

Pass 1: 33 open codes, 1 excluded. Pass 2: 25 codes, 2 excluded, 22 apps, max 2
per app. No reviewId overlap. 18 of pass 1's 50 carry 4–5 stars.

## RQ2 — Accuracy vs interface (M1, n = 324,687, all q < 0.05)
| Complaint | β | 95% CI |
|---|---|---|
| Intrusive ads | **−1.439** | [−1.460, −1.418] |
| Prayer-time accuracy | **−1.112** | [−1.142, −1.083] |
| Stability / bugs | −0.938 | [−0.965, −0.910] |
| Reminders / adhan | −0.845 | [−0.868, −0.823] |
| UI design | −0.600 | [−0.664, −0.537] |
| Calculation method | −0.600 | [−0.769, −0.430] |
| Qibla | −0.550 | [−0.594, −0.507] |
| Madhab | −0.324 | [−0.526, −0.122] |

- **Accuracy-family mean −0.647** vs interface −0.600 — report *with* the spread
- `madhab` ordinal check: β = +0.216, p = 0.649 — **sign flip**
- **Star masking:** of 7,942 reviews with a negative accuracy complaint,
  **3,306 (41.6%) gave 4–5 stars; 1,830 (23.0%) gave 5**
  (1★ 2,319 · 2★ 914 · 3★ 1,403 · 4★ 1,476 · 5★ 1,830)
- Qibla failure mode: **compass instability**, not miscalculation

## RQ3 — Travel
| | | |
|---|---|---|
| Mosque finder, cluster-robust | β = −0.137, p = 0.003, 21 clusters, n = 3,010 | ✅ |
| Mosque finder, mixed model | +0.05, **NaN SE — cannot arbitrate** | ✅ |
| Apps *with* feature | 1,421 mentions / 10 apps — 27.0% neg, 38.6% pos | ✅ |
| Apps *without* | 1,589 mentions / 11 apps — 23.0% neg, 43.6% pos | ✅ |
| Mosque-finder demand where absent | 243 signals, 9 apps | ✅ |
| Preference citations | 4,696 total | ✅ |
| — aspects cited | adhan 374 · Quran audio 293 · accuracy 290 · qibla 155 · adhkar 152 | ✅ |
| — citation rate | 1.38% with mosque finder vs 1.24% without | ✅ |
| Qasr: implementing apps | **1 of 26** (Muslim Bangla) | ✅ |
| Qasr demand | 21 signals; 20 from 5 apps lacking it | ✅ |

## RQ4 — Inclusion
| | | |
|---|---|---|
| Tagged mentions | 2,194 across 24 of 26 apps — **20% precision** | ⚠️ |
| Adjusted estimate | **~440 genuine** | ⚠️ |
| Apps with the feature | **3** (Athan, Islamic Habit Tracker, Pillars) | ✅ |
| Sentiment, *with* feature | 26.3% negative (464 mentions, 3 apps) | ✅ |
| Sentiment, *without* | 26.0% negative (1,718 mentions, 21 apps) | ✅ |
| Demand | 272 total; **216 from 16 apps lacking it** | ✅ |
| — types | request 161 · preference 51 · churn 36 · absence 21 · tenure 3 | ✅ |
| Δ star, review-weighted (M3) | **+0.085** [+0.074, +0.096] — NOT ESTIMABLE | ✅ |
| Δ star, app-level unweighted | −0.236 — **opposite sign, same data** | ✅ |

## RQ5 — Bloat and hardware
| | | |
|---|---|---|
| feature_count → bloat | β = −0.000487, p = 0.272 (**cluster-robust**) | ✅ |
| M4 `feature_count` | **DO NOT CITE** — SE 743.8, p ≈ 0.99999 | ✅ |
| Co-occurrence, all | lift **23.98×**, OR 29.41, p = 7.4e-137, n = 324,687 | ✅ |
| Co-occurrence, separate sentences | lift **7.62×**, OR 8.78, p = 9.3e-24, n = 324,593 | ✅ |
| Fused | 94 of 134 reviews (70%) fuse; 40 (30%) separate | ✅ |
| 5 leanest (6–9 features) | 3.96 stars, 5.2% bloat rate | ✅ |
| 5 fullest (13–14) | 4.71 stars, 1.6% bloat rate | ✅ |
| Robustness (drop n=5, n=45) | 4.20 stars, 5.6% | ✅ |
| Cleanest pair | iPray 8 feat @ 8.0% vs Athan 13 feat @ 1.5% | ✅ |
| features ~ installs | Spearman ρ = 0.244, n = 26 | ✅ |
| App-level (exploratory) | ρ = +0.471 sentiment (p = .015), +0.422 score (p = .032) | ✅ |
| Hardware, inside iQIBLA | 37.7% neg, 45.9% pos (n = 518) | ✅ |
| Hardware, other 23 apps | 49.2% neg, 20.8% pos (n = 1,994) | ✅ |
| Hardware mentions | **2,512** tagged — **40% precision** → ~1,004 | ✅ resolved: 518 + 1,994 = 2,512; the old 2,482 was stale |
| Corpus mean star | 4.40 | ✅ |
| `rq5_bloat` stability | 6–11 topics over 5 seeds, ARI 0.452 | 📋 |
| `rq1_tracker` stability | ARI 0.837 | 📋 |

## RQ6 — Gaps
| | | |
|---|---|---|
| Promised pairs | 272; 136 judgeable at n ≥ 30 | ✅ |
| Broken promises | **18 across 11 apps** (>40% negative) | ✅ |
| Most broken by share | monetization 5/16 (31%) · adhan 5/20 (25%) | ✅ |
| Largest by volume | Muslim Pro adhan 9,989 @ 45.6%; monetization 9,182 @ 54.8% | ✅ |
| Served well | madhab, monetization, qibla, table_format, widgets | ✅ |
| Coverage < 25% | calendar_sync, companion_hardware, qasr_travel, women_period | ✅ |
| ⚠️ `tracker_score` | 1 of 1 broken — **one app, ratio meaningless** | ✅ |

**Unmet-need ranking** (demand × apps lacking) — built from demand signals (91.5%
precision), not raw aspect counts:
| Aspect | Score | Apps lacking |
|---|---|---|
| companion_hardware | 6,725 | 25 |
| women_period | 4,738 | 23 |
| mosque_finder | 3,304 | 14 |
| forbidden_times | 2,805 | 15 |
| nafl_times | 2,445 | 15 |
| prayer_tracker | 1,905 | 15 |

## Topic model
| | | |
|---|---|---|
| Topics | 40 + outlier bucket | ✅ |
| Topic 0 (generic praise) | 168,054 (61%) | ✅ |
| Outliers | 62,745 (22.7%) | ✅ |
| Interpretable coverage | ~45,600 docs (~16%) across 39 topics | ✅ |
| Hand-labelled | 67 topics across 4 sheets | ✅ |
| RQ4 sub-model collapse | 6 of 10 topics on one theme, **58%** of non-outlier docs | ✅ |

## Never write these
| Wrong | Right |
|---|---|
| "record integrity 52%" | C at ~21%; A and B are app-specific |
| "no tier effect (M2)" | cite the cluster-robust OLS |
| "the accuracy coefficient is −1.112" | that is prayer-time accuracy, one of four |
| "p = 0" for the dip test | p < 1e-16 |
| "636 qasr mentions" | 21 demand signals |
| "3 models not estimable" | 11 cannot support inference |
| "tracker_score most broken" | 1 of 1 app; report adhan and monetization |
| "2,194 menstrual mentions" | ~440 genuine (20% precision) |
| pooled coding percentages | two passes, separately |
| RQ4 Δ = −0.236 | M3 records +0.085; sign flips under reweighting |
| "40 topics" implying corpus coverage | ~16% of modelled documents |
