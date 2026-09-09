# Disclosures — what the paper must say out loud

Every item here is something a reader could otherwise be misled about. Each one
has a **claim ceiling** (what the evidence will actually support) and, where it
helps, a sentence you can adapt. Work through it once while drafting and once
before submission.

Two provenance markers are used throughout:

- ✅ **verified** — checked on 2026-09-07 against the outputs of the analysis run of **2026-09-03** (`model_results.csv`, the figures and the gap matrix all carry that timestamp; the pipeline was not re-run afterwards, only the notebooks that read it)
- 📋 **from the run** — produced by the pipeline on the analysis server and
  carried over from `MANUAL_WORK_GUIDE.md`; not independently re-derived here

---

## 1. Sampling and qualitative evidence

### 1.1 Two coding passes, and they must be reported separately ✅

**Pass 1 — 50 reviews, one app** (*Athan: Prayer Times & Al Quran*). Selection
artefact, not a corpus property: `06_models.py` takes `nsmallest(50, "sent_num")`,
`sent_num` is three-valued, so every negative review ties at −1 and pandas
returns the first fifty in frame order — app order.

**Pass 2 — 30 reviews, 22 apps** (`_draw_tracker_extension.py`): a floor of one
per app with the remainder allocated round-robin, so no app contributes more than
two. Already-coded reviewIds excluded, seeded. Eligible pool 1,173 — which proves
pass 1 was single-app through the tie-break alone, not because one app dominates
tracker complaints.

**Do not pool the two passes.** Their designs differ — 50 from one app against 30
spread across 22 — so a pooled percentage would silently re-weight the ecosystem
toward Athan. Report them side by side and let the comparison do the work.

> Two coding passes were conducted. The first covered fifty reviews from a single
> application, selected before we identified a tie-break artefact in the sampling
> rule. The second drew thirty reviews stratified across the remaining
> twenty-two tracker applications and was coded against the frame derived from
> the first. We report the passes separately, since their sampling designs differ.

### 1.2 "Negative" means sentiment, not stars ✅

18 of the 50 carry 4–5 star ratings. Selection is by RoBERTa sentiment on the
review text, so the set includes otherwise-positive reviews containing one
specific complaint.

**Write:** "reviews expressing tracker dissatisfaction" — not "negative reviews".

### 1.3 The frame, and what the second pass did to it ✅

Pass 1 produced 33 open codes over 50 reviews, consolidated into six themes plus
an exclusion. Pass 2 produced 25 codes over 30 reviews, coded against that frame,
and needed **two new themes**. In both passes every code maps to exactly one
theme — no code is split, which is what makes the frame reportable.

Excluded rows are reported, not dropped: 1 in pass 1 (n = 49), 2 in pass 2
(n = 28).

| Theme | Pass 1 (Athan, n=49) | Pass 2 (22 apps, n=28) |
|---|---|---|
| C. The record can't be trusted | 11 (22%) | 6 (21%) |
| D. Normative mismatch | 6 (12%) | 3 (11%) |
| E. Prompting failures break the loop | 7 (14%) | **13 (46%)** |
| A. Redesign as regression | 10 (20%) | **0** |
| B. Retroactive logging denied | 5 (10%) | **0** |
| F. Monetization against religious purpose | 10 (20%) | 1 (4%) |
| G. The record is gated, not owned | — | 3 (11%) *new* |
| H. App overreaches into the device | — | 2 (7%) *new* |

**This table is a result, not bookkeeping.** Read it in the paper as follows.

- **C and D replicate almost exactly** across two independent samples. They are
  ecosystem-level findings and can be stated as such.
- **E triples.** Prompting failure — location detection, notifications, adhan
  playback, calendar drift — is the dominant tracker complaint ecosystem-wide and
  was under-represented in the single app.
- **A and B vanish entirely.** They were one application's bad redesign, not a
  property of prayer tracking. Had pass 1 been published alone, 15 of its 49
  complaints would have been presented as a general finding about retroactive
  logging and redesign.
- **F collapses from 20% to 4%**, so monetization grievance was also largely
  app-specific.
- **G and H exist only outside Athan**: access to one's own record gated behind a
  login or paywall, and apps demanding device-level settings changes.

---

## 2. Topic modelling

### 2.1 The topic model does not characterise the corpus ✅

Of 276,422 modelled documents, 168,054 sit in topic 0 (generic praise, no finer
structure) and 62,745 are outliers. **About 83% of documents are therefore in
either a non-topic or a non-finding**, leaving roughly 45,600 across the 39
interpretable topics.

**Claim ceiling:** the topics describe the *interpretable minority* of reviews.
Never imply 40 topics cover the dataset.

> Reduced to forty topics, the model assigned 22.7% of documents to the outlier
> bucket and a further 61% to a single undifferentiated praise cluster. The
> remaining thirty-nine topics, which we label and report, cover approximately
> 16% of modelled documents.

### 2.2 Topic structure varies by seed 📋

`rq5_bloat` produced 11, 10, 7, 11 and 6 topics across five fixed seeds, mean
pairwise ARI **0.452**. `rq1_tracker` is stable at ARI **0.837**.

**Claim the themes, not the partition.** Write "bloat complaints cluster around
legibility, speed and advertising", never "we identified exactly ten bloat
topics". Report the ARI and the 6–11 range in Methods.

### 2.3 The RQ4 sub-model collapsed ✅

**Five of the ten topics carry the identical label** `Data selling to US military
complaint`, and a sixth (`Complaint about selling data`) is a near-duplicate — so
**six of ten topics sit on one theme, holding 58% of the sub-model's non-outlier
documents** (4,788 of 8,285). The Muslim Pro data-selling story swamps the subset.

> Corrected 2026-09-09. An earlier count said "six of ten plus a seventh, 63%".
> That treated the outlier bucket as a topic — it had carried a content label
> before being relabelled `OUTLIERS`, which inflated both the topic count and the
> document share. Use 6 of 10 and 58%.

**Claim ceiling:** themes only. Do not present ten topics as ten findings.

### 2.4 Topic −1 is not a topic ✅

The outlier bucket is labelled `OUTLIERS` in all four sheets. It must never
appear in a results table as a theme.

---

## 3. Measurement and precision

### 3.1 Aspect precision is estimated on a subset 📋

**84.4% [79.7, 88.9]** corpus-weighted, **69.3%** unweighted, measured on **192 of
300** gold rows — those the current configuration still produces. The gold set
was labelled against an earlier, broader lexicon; every change since only removes
matches, so the surviving rows are an unbiased subset rather than a re-labelling.
Say this explicitly; a reader who sees "192 of 300" without it will assume
cherry-picking.

### 3.1a Per-aspect precision — SEVEN aspects fall below 50% ⚠️✅

**Verified 2026-09-09 by running `_aspect_precision.py`.** The headline figures
reproduce exactly (69.3% unweighted, 84.4% [79.7, 88.9] weighted, 192 of 300 gold
rows surviving). **But the weighted figure is dominated by high-volume aspects,
and several low-volume aspects the paper reports directly are near-useless.**

| Aspect | corpus n | gold | precision | 95% CI | implied true |
|---|---|---|---|---|---|
| reminders_adhan | 30,069 | 7 | 100% | [65, 100] | 30,069 |
| quran_audio | 24,964 | 6 | 100% | [61, 100] | 24,964 |
| prayer_times_accuracy | 21,941 | 11 | 73% | [43, 90] | 16,016 |
| ads_intrusive | 17,845 | 11 | 100% | [74, 100] | 17,845 |
| ui_design | 15,720 | 11 | 73% | [43, 90] | 11,475 |
| monetization | 13,914 | 11 | 82% | [52, 95] | 11,409 |
| qibla | 8,526 | 11 | 82% | [52, 95] | 6,991 |
| stability_bugs | 7,665 | 11 | 64% | [35, 85] | 4,905 |
| privacy_data | 7,471 | 10 | 90% | [60, 98] | 6,723 |
| prayer_tracker | 6,009 | 9 | 78% | [45, 94] | 4,687 |
| complexity_bloat | 5,389 | 9 | 67% | [35, 88] | 3,610 |
| widgets | 4,755 | 11 | 100% | [74, 100] | 4,755 |
| **mosque_finder** | 3,043 | 11 | **27%** | [10, 57] | **~821** |
| **companion_hardware** | 2,512 | 5 | **40%** | [12, 77] | **~1,004** |
| **women_period** | 2,194 | 10 | **20%** | [6, 51] | **~438** |
| **forbidden_times** | 1,534 | 10 | **10%** | [2, 40] | **~153** |
| **table_format** | 1,483 | 9 | **44%** | [19, 73] | **~652** |
| calc_method | 639 | 10 | 90% | [60, 98] | 575 |
| **qasr_travel** | 637 | 2 | **0%** | [0, 66] | **~0** |
| madhab | 491 | 5 | 60% | [23, 88] | 294 |
| **tracker_score** | 101 | 1 | **0%** | [0, 79] | **~0** |

**Which claims this touches, and what to do:**

- **RQ4** — "2,194 mentions across 24 apps" is the headline. At 20% precision the
  true figure is nearer **438 [124, 1,118]**. **Do not report 2,194 as a count of
  genuine menstrual-handling mentions.** RQ4's argument survives because it rests
  on *demand signals* (272, validated separately at 91.5% — §3.3) and on the
  3-of-26 feature scarcity, neither of which depends on this aspect's precision.
  Rewrite the volume sentence; keep the argument.
- **RQ3** — `mosque_finder` at 27% means "3,043 mentions" is nearer **821**. The
  27.0%-vs-23.0% negative-sentiment comparison is computed over noisy tags on
  *both* sides, so the direction may survive but the precision must be stated.
- **RQ5** — `companion_hardware` at 40%: 2,512 mentions is nearer **1,004**, and
  the 37.7%-vs-49.2% split rests on tags of this quality.
- **RQ6** — the unmet-needs ranking multiplies demand volume by apps lacking.
  **Demand signals are separately validated at 91.5%**, so the ranking itself is
  on firmer ground than the raw mention counts. Say which input is which.
- **`qasr_travel` at 0 of 2** confirms §4.4 quantitatively. Report the 21 demand
  signals; **never** report 637 as qasr discussion volume.
- **`tracker_score` at 0 of 1** — a single gold row. Uninformative either way; do
  not lean on it, and note the 1-of-1 broken promise in §4.7 rests on the same
  thin evidence.

⚠️ **Rule for the whole paper: any aspect-level count must carry its precision**,
and for the seven aspects above 50% is not achieved, so a bare count would
mislead. Report either the precision-adjusted estimate with its interval, or the
raw count explicitly labelled as tagger output rather than as verified mentions.

**Gold n per aspect is 1–11**, so these intervals are wide. They are the honest
uncertainty, not a reason to prefer the point estimate.

**This is a measurement limitation, not a faulty run.** Nothing needs
re-executing. Raising precision would mean narrowing the lexicon for these
aspects and re-running `03_absa` and everything downstream — a multi-day cascade,
and future work rather than a pre-deadline fix.

### 3.2 Two aspects have no gold rows 📋

`calendar_sync` and `goal_system` (0.2% of corpus volume) are excluded from the
weighted figure.

### 3.3 Demand-pattern precision — figures verified, but the sample is stranded ⚠️

**Verified 2026-09-09** against `demand_precision_sample.backup-20260903-210822.csv`,
which holds all 200 hand labels:

| request_type | precision | n |
|---|---|---|
| preference | 100.0% | 40 |
| tenure | 100.0% | 40 |
| churn | 97.5% | 40 |
| request | 95.0% | 40 |
| **absence** | **65.0%** | 40 |
| **overall** | **91.5%** (183/200) | 200 |

Both documented figures reproduce exactly. **But there is a provenance problem
that must be resolved before submission.**

`03c_demand_mining.py` draws this sheet with `stratified_sample(..., seed=0)` —
deterministic *given its input*. After the 200 rows were labelled, an upstream
threshold changed, `03c` was re-run, and the new demand output produced a
**different** 200-row sample. `write_gold_sheet` did its job: it backed the
labelled file up and warned that the regenerated sheet is a new sample needing a
merge on `reviewId`. The merge recovered only what could match — **11 of 200**.

So today: `demand_precision_sample.csv` holds 11 labels, its backup holds 200,
and only 12 reviewIds are common to both. **The 91.5% figure therefore describes
a superseded version of the demand extraction, while the demand counts the paper
reports (272 for `women_period`, 243 for `mosque_finder`, 269 for
`companion_hardware`, 21 for `qasr_travel`) come from the current one.**

⚠️ This is *not* the same situation as §3.1. There the argument is that lexicon
changes only ever *remove* matches, so surviving gold rows are an unbiased subset.
Here the extraction changed in a way that produced a different sample and the
direction is unknown, so no equivalent argument is available.

**The fix: re-label the current 200-row sheet** (`src/data/gold_labels/demand_precision_sample.csv`),
which is already stratified 40 per `request_type`. Roughly one to two hours. Then
precision describes the data actually reported. Keep the backup committed either
way — it is 200 rows of irreplaceable hand work.

If the deadline forbids re-labelling, the fallback is to report 91.5% while
stating plainly that it was measured on a sample drawn before the final demand
run. That is defensible but weaker, and `absence` at 65% is already the figure a
reviewer will press on.

### 3.4 Promise-source agreement is low ✅

Mean κ = **0.272** ✅ verified against `promise_source_agreement.csv`. Per-feature
κ ranges from 0.62 (Women tracking) down to 0.36 (Companion hardware); the CSV
also records where the store description *misses* a feature the CSV records and
where it *overclaims*. This is weak agreement and must be stated as such wherever
promise data is used.

### 3.5 The zero-shot backstop was dropped 📋

Measured **11.5%** precision on 300 hand-labelled sentences, so it was removed.
Reporting a discarded component is evidence of calibration, not weakness.

### 3.6 Devotional formulae were filtered — this is a contribution 📋

1,686 sentences (0.8%) removed before aspect tagging. Generic English feature
vocabulary misfires badly on religious register: `goal_system` scored **0%**
precision because reviewers write "May Allah reward you". Frame this as a finding
about applying off-the-shelf NLP to devotional text, not as a cleaning step.

### 3.7 Inter-rater agreement ✅

Krippendorff's α = **0.824** — **recomputed independently 2026-09-09** from
`doc_500_annotator1.csv` and `doc_500_annotator2.csv`: 100 overlap rows, 89.0%
raw agreement, nominal α = 0.824 over categories {Positive, Neutral, Negative,
Mixed}. Reproduces the documented figure exactly.

---

## 4. Statistics

### 4.1 Eleven coefficients cannot support inference, not three ✅

**Verified 2026-09-07 against `model_results.csv` from the analysis run of 2026-09-03.** "Not estimable"
and "no effect" are different claims, and the file contains two distinct kinds of
unusable row:

- **Three with no standard error at all** — M3 (`women_feature_score_diff`), M5
  (`has_mosque_finder`), M5-demand. These are the ones already known about.
- **Eight with standard errors that exploded**, all from the primary mixed
  models: M2's two tracker-tier terms (SE 2.0e6 and 1.3e6) and six M4 terms
  (SE 4.1e2 to 5.6e3). Every one reports p ≈ 0.99999.

The eight are the dangerous ones, because they **carry q-values and sit inside
the BH denominator of 29**, so they read as "tested, not significant" when they
are actually "failed to converge". A random intercept per app over 26 apps with
app-level predictors is the same structural problem the code comments already
note for M4.

**Never write "no effect" for any of these eleven.** Where a null is genuinely
wanted, take it from the cluster-robust OLS (`M4-sens`), which converged.

**20 of 29 coefficients survive BH-FDR at q < 0.05** ✅ — this count is confirmed
exactly. Report the full denominator, and say that eight of the 29 are
non-converged terms retained in the correction for transparency.

### 4.2 RQ1 — the null is the finding 📋

✅ **Corrected against the run.** The tier null must be cited from the
cluster-robust OLS, **not** from the M2 mixed model. M2's two tier terms are
degenerate — estimates of 13.66 and 13.87 with standard errors of 2.0e6 and
1.3e6, p ≈ 0.99999. The familiar p = 0.49 and p = 0.35 are the `M4-sens`
cluster-robust rows (0.488 and 0.349), which converged. Citing M2 as "no
difference" is exactly the error §4.1 warns against.

So: no difference in sentiment by tracker tier (p = 0.49, p = 0.35,
cluster-robust OLS on 25 app clusters; the mixed model did not converge).
Sentiment is bimodal (Hartigan's D = 0.118 ✅ confirmed; the reported p
underflows to 0, so write p < 1e-16, never "p = 0"). Guilt language is
**under-represented**: Fisher OR = 0.28, p = 3.5e-11 ✅ confirmed, n = 149 — the
*opposite* of the streak-anxiety hypothesis.

✅ **The qualitative frame corroborates this across two independent samples,
and sharpens it.** Do not conflate the constructs below — the distinction is the
contribution.

**Guilt is rare in both passes: 3 of 77 coded complaints (4%).** Pass 1 gave
`Tracker framing feels judgmental` and `Imposed goals replace own tracking`;
pass 2 gave `Reminders induce guilt`. A hypothesis this prominent in the
literature surviving at 4% across two samples, and contradicted by the Fisher
test at OR = 0.28, is a robust null.

**What dissatisfaction is actually about, ecosystem-wide:**

- **Prompting failure (E) — 46% of the cross-app pass.** The tracker fails
  because the app never reliably prompts: location detection failing, adhan not
  playing, notifications lost, calendars drifting. The loop breaks before
  logging is even at issue.
- **Record trustworthiness (C) — ~21% in both passes.** Data lost, counts
  regressing, the tracker silently disabling itself.
- **Normative mismatch (D) — ~11% in both passes.** The tracker mis-models
  religious practice: menstrual exemption absent, no sunnah support, imposed
  goals displacing the user's own. **Not** guilt — a mismatch between the app's
  model of worship and the user's.

**The claim to make:** tracker dissatisfaction is overwhelmingly about the app
failing to prompt and failing to keep a trustworthy record. Where a normative
complaint appears it concerns the app's model of worship rather than pressure to
perform. This *reframes* the streak-anxiety literature rather than merely nulling
it, and both the statistics and two independent coding passes point the same way.

**What you must not claim.** An earlier reading of pass 1 alone put "record
integrity" at 52%. That figure does not survive: 15 of those 26 complaints were
themes A and B, which are absent from all 22 other apps. Report C at ~21%, and
report A and B as an application-specific redesign episode.

### 4.2a RQ1 carries direct RQ4 evidence ✅

Menstrual-exemption complaints appear in **both** passes — 3 in pass 1
(`Menstrual mode unreliable`), 1 in pass 2 (`Menstrual exemption absent`), the
latter from a different application. RQ4 is otherwise not estimable at 3 apps, so
this is rare first-hand evidence that bio-spiritual inclusion fails in practice
and not merely in availability. Cite in RQ4 and cross-reference RQ1.

### 4.2b The two passes are a methods contribution ✅

Themes A, B and F accounted for 51% of the single-app pass and 4% of the
cross-app pass. That is a concrete demonstration that app-level qualitative
sampling in app-store research can manufacture themes which look like properties
of a genre and are properties of one release. Worth a short methods paragraph;
it costs three sentences and pre-empts the obvious reviewer question about how
the sample was drawn.

### 4.3 RQ2 — the family mean holds; one member does not ✅

**Verified against `model_results.csv`.** The guide's figures are correct:
accuracy-family mean **−0.647**, interface **−0.600**. RQ2's contrast is the
*family* — `prayer_times_accuracy`, `qibla`, `madhab`, `calc_method` — against
`ui_design`, as the notebook header states. At that level the two are close and
the original instruction stands: **do not claim accuracy wins as a class.**

But the family average conceals a wide spread, and the paper should say so:

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

**Prayer-time accuracy alone costs −1.112, with an interval disjoint from
interface.** The family mean is dragged to −0.647 by `madhab` (−0.324) and
`qibla` (−0.550), which are rarer and milder. So:

> As a class, accuracy complaints cost no more than interface complaints
> (−0.647 against −0.600). That average conceals a wide spread: prayer-time
> accuracy alone costs −1.112 [−1.142, −1.083], nearly double the interface
> penalty and with a non-overlapping interval, while madhab and qibla complaints
> cost markedly less.

**Do not report the family mean without the spread**, and do not report −1.112
as "the accuracy coefficient" — it is one member of four.

**Star ratings hide accuracy complaints** ✅ (derived 2026-09-07 from the 2026-09-03 run). Of 7,942
reviews carrying at least one negative accuracy-family complaint, **3,306 (41.6%)
gave four or five stars and 1,830 (23.0%) gave five**. Star distribution:
1★ 2,319 · 2★ 914 · 3★ 1,403 · 4★ 1,476 · 5★ 1,830. This is a finding, not a
caveat — it means the rating signal systematically under-represents accuracy
defects, and it is the strongest thing in RQ2.

**The dominant qibla failure is instability, not miscalculation** ✅. Topic 9
(`Inaccurate Qibla direction`, 1,771) outweighs topic 17 (`Inaccurate Qibla
compass pointer`, 698), but topic 9's representative reviews describe the compass
giving different bearings from one spot as the phone rotates. Report it as a
sensor/calibration problem; do not write that apps compute the wrong bearing.

**One thing the family framing hides entirely: intrusive advertising is the
costliest complaint in the corpus at −1.439**, above every accuracy term. It ties
RQ2 to RQ5's ad-heavy bloat topics and to `Monetization against religious
purpose` in the RQ1 coding. Advertising runs through three research questions.

`madhab` still flips sign in the ordinal check (β = +0.216, p = 0.649 ✅
confirmed), so keep that caveat.

### 4.4 RQ3 — the sign flip has an explanation, and a lexicon caveat ✅

Mosque finder β = **−0.137** (p = 0.003, cluster-robust, 21 clusters, n = 3,010)
✅ confirmed. The mixed model gives +0.05 with a NaN standard error ✅ — it cannot
arbitrate, and the reversal must be stated.

**Why the sign runs that way** ✅ (derived 2026-09-07 from the 2026-09-03 run): apps *with* the feature
show 27.0% negative mentions against 23.0% without, positives 38.6% against
43.6%. The groups are performing different speech acts — the 11 apps without it
generate 243 demand signals across 9 apps (requests, which read positive), while
the 10 apps with it generate usage reports. Report the gap as wishing-versus-using,
never as the feature harming apps.

**Travel is not a differentiator** ✅. Of 4,696 preference citations, the named
aspects are adhan reminders (374), Quran audio (293), prayer-time accuracy (290),
qibla (155), adhkar (152). Mosque finder is not in the top eight; qasr is absent.
Citation rates: 1.38% with a mosque finder against 1.24% without.

⚠️ **The `qasr_travel` aspect over-captures.** Its 636 mentions across 18 apps run
76.7% positive and 7.1% negative, implausible for a feature 25 of 26 apps lack,
and only 3.3% of mentions are demand signals. The lexicon is catching travel and
pilgrimage talk generally. **Do not report 636 as qasr discussion volume.** Rest
the qasr argument on the 21 demand signals, which were read directly. This belongs
in the precision discussion alongside §3.6.

**Qasr is N = 1** — one implementing app. No between-app estimate; say so.

### 4.5 RQ4 — not estimable, and the feature does not help ✅

**2,182 distinct reviews** (2,194 mentions) across **24 of 26 apps**, 0.672% of the
analysed corpus. Only **3 apps** ship the feature: Athan, Islamic Habit Tracker,
Pillars. No between-app estimate is possible — say so plainly. M3 returns no
standard error (`women_feature_score_diff` 0.085, flagged underpowered).

**Sentiment is identical with and without the feature** ✅ (derived 2026-09-07 from the 2026-09-03 run):
**26.3% negative** in the three apps that ship it against **26.0%** in the 21 that
do not. Having menstrual handling is not associated with users being happier about
menstrual handling. Combined with §4.2a, the claim strengthens from "almost nobody
builds this" to "almost nobody builds it, and the implementations that exist are
not meeting the need."

**Demand** ✅: 272 signals, **216 from the 16 apps lacking the feature** — 161
requests, 51 preference citations, 36 churn statements, 21 explicit absences.

**Δ star: the sign depends on the aggregation** ✅ — which is itself the argument
that it is not estimable.

| Aggregation | Δ | Detail |
|---|---|---|
| Review-weighted (**M3, the recorded figure**) | **+0.085** [+0.074, +0.096] | 4.476 vs 4.391; 46,741 reviews / 3 apps vs 277,946 / 23 |
| App-level unweighted | **−0.236** | 4.135 vs 4.371 |

Athan alone is 44,954 of the 46,741 treated reviews, so the weighted figure is
close to a single application; the unweighted one is pulled down by Islamic Habit
Tracker, which §9.3 names as carrying the corpus's lowest rating. **Report M3's
+0.085, say it is not estimable, and mention the reversal** — a difference that
changes sign under reweighting is the cleanest demonstration that N = 3 cannot
answer this. M3 carries no standard error. Combine with §2.3.

### 4.6 RQ5 — lead with the fused cases, report the bound ✅

Co-occurrence lift **23.98×** (OR 29.41, p = 7.4e-137) across all co-occurring
reviews; **7.62×** (OR 8.78, p = 9.3e-24) restricted to separate sentences, with
the 94 fused reviews removed ✅ all confirmed against `model_results.csv`. The two
lexicons share no keyword and no regex, so a fused sentence carries two distinct
terms. Report 7.62× as the conservative bound.

**The feature-count null must be cited from the cluster-robust fit** ✅. M4's
`feature_count` is degenerate (β = −0.004, SE 743.8, p ≈ 0.99999). The converged
estimate is `complaint_complexity_bloat × feature_count` = −0.000487, p = 0.272.

**The relationship inverts** ✅ (derived 2026-09-07 from the 2026-09-03 run). Five leanest apps (6–9
features): 3.96 mean stars, 5.2% bloat-complaint rate. Five fullest (13–14): 4.71
stars, 1.6%. Corpus mean 4.40. Robust to dropping the two smallest apps (n = 5,
n = 45), which leaves 4.20 stars at 5.6%. iPray (8 features, 2,225 reviews) draws
bloat complaints at 8.0% against Athan (13 features, 44,954 reviews) at 1.5%.
Feature count and installs correlate only weakly (Spearman ρ = 0.244, n = 26), so
popularity does not explain the null.

**Companion hardware is N = 1 and reads counterintuitively** ✅. Inside iQIBLA
Life: 37.7% negative, 45.9% positive (n = 518 mentions). In the 23 apps without
hardware: 49.2% negative, 20.8% positive (n = 1,994) — largely unmet requests for
watch support. Shipping hardware is associated with *better* hardware sentiment.
2,482 mentions across 24 apps. Descriptive only; no between-app estimate.

**Claim themes, not the partition** — 6–11 topics across five seeds, ARI 0.452.

### 4.7 RQ6 — two different gaps ✅

**Delivery gap.** 272 promised (app, feature) pairs, 136 judgeable at n ≥ 30,
**18 broken promises across 11 apps** (>40% negative on a promised feature). By
share of promising apps that break it: monetization 5/16 (31%), reminders/adhan
5/20 (25%), mosque finder 1/4, forbidden times 1/4, widgets 2/13, prayer-time
accuracy 2/19, qibla 1/18. Largest by volume: Muslim Pro reminders/adhan (9,989
mentions, 45.6% negative) and monetization (9,182, 54.8%).

⚠️ `tracker_score` shows 1 of 1 broken. **Do not report it as the most-broken
feature** — a single promising app makes the ratio meaningless.

**The two most-broken features are the two most basic**: adhan reminders (the core
function) and monetization (the terms it is offered on). Failure concentrates in
the promise the app is built around, not in advanced capability.

**Coverage gap.** Unmet-need score (demand × apps lacking): companion hardware
6,725 (25 lacking), women_period 4,738 (23), mosque finder 3,304 (14), forbidden
times 2,805 (15), nafl times 2,445 (15). Four aspects under 25% coverage:
`calendar_sync`, `companion_hardware`, `qasr_travel`, `women_period`.

**Served well** (>50% coverage, <30% mean negative): madhab, monetization, qibla,
table_format, widgets. **Monetization appears on both lists** — not a
contradiction: a 31% break rate against a 25.8% mean negative rate is most apps
handling payment acceptably and a minority handling it badly. Explain it rather
than letting a reviewer find it.

---

## 5. Provenance — the pre-registration sentence

Two different dates, and conflating them would misrepresent the work.

- The **hypothesis and test** for RQ5 were pre-specified on **2026-08-26**, before
  the full run. That is the pre-registration and it stands.
- The **choice of reporting emphasis** (Option B — leading with the fused cases)
  was made on **2026-09-07**, with the co-occurrence figures and the labelled
  topics already visible.

> The co-occurrence hypothesis and its test were specified before the analysis
> was run. The decision to foreground the fused cases was made after inspecting
> the results, and we report the separate-sentence estimate as a conservative
> bound.

**Do not describe the framing itself as pre-specified.**

---

## 6. Reproducibility notes

- **Seeds are fixed.** UMAP is seeded; the stability pass uses seeds 42–46,
  chosen in advance so the set could not be adjusted after seeing which one
  flattered a result. Say that.
- **Ship the server's `run_manifest.jsonl`** with the artefact submission — it
  records commit, GPU, batch size and library versions for every reported number.
- ✅ **Reproducibility confirmed 2026-09-09.** `06_models.py` was re-run from a
  clean tree at commit **`5da2db7`** and `model_results.csv` came back byte-identical
  to the 2026-09-03 output except for one note string (`"definitional same-sentence
  pairs removed"` → `"same-sentence pairs removed"`, from commit `9c5b48b`). Every
  coefficient, p-value and n reproduced. **Cite `5da2db7` in Methods** — the
  manifest's earlier `ff103ae` was a server-local commit that was never pushed and
  resolves nowhere.
- ⚠️ **Cite the analysis run as 2026-09-03**, not 2026-09-07. `model_results.csv`,
  `gap_matrix.csv` and all fourteen figures carry that timestamp. The pipeline was
  not re-run afterwards; on 2026-09-07 only the notebooks that *read* those outputs
  were executed, plus three ad-hoc query scripts which appear in the manifest and
  produced no reported figure of their own. Take the commit hash for Methods from
  the manifest entry for `06_models.py` on 2026-09-03, not from the later entries.
- ⚠️ **The manifest's later entries show `dirty_working_tree: true`.** Those are the
  ad-hoc queries, not the analysis. Say which entry the paper's numbers come from.
- ✅ **A defect was found and fixed during analysis:** sub-topic models shared a
  hardcoded output path and overwrote the ecosystem topic sheet on every run. The
  sheet was rebuilt from the saved model and the assignment record without
  refitting. No statistic depended on it. Worth a line in a reproducibility
  appendix if you have one — finding and reporting it is a credibility gain.
- ✅ **Representative documents in the rebuilt sheet** were selected by topic
  probability rather than BERTopic's own c-TF-IDF ranking, because the saved
  model does not serialise `representative_docs_`. Deterministic, and labels were
  read against the full document set in the notebook regardless.

---

## 7. Corpus figures 📋

| Measure | Value |
|---|---|
| Corpus | 732,194 reviews, 26 apps |
| Analysed (has text) | 324,687 |
| Topic-modelled (C_en) | 276,422 in 40 topics |
| Outlier bucket | 62,745 (22.7%) ✅ |
| Topic 0, generic praise | 168,054 (61%) ✅ |
| Topics labelled by hand | 67 across four sheets ✅ |
| Krippendorff's α | 0.824 ✅ |
| Aspect precision, weighted | 84.4% [79.7, 88.9] |
| Coefficients surviving q<0.05 | 20 of 29 |
| Cannot support inference | 11 (3 no SE + 8 non-converged) ✅ |

---

## 8. Final pass before submission

- [ ] Every ✅ item above appears somewhere in the paper
- [ ] No sentence describes the RQ5 *framing* as pre-specified
- [ ] No "not estimable" model is described as a null result
- [ ] No claim that accuracy costs more than interface
- [ ] Both coding passes are reported, separately, never pooled
- [ ] Themes A and B are described as application-specific, not general
- [ ] No "record integrity 52%" figure survives anywhere in the draft
- [ ] The accuracy-family mean (−0.647) is reported with its spread, never alone
- [ ] −1.112 is labelled as prayer-time accuracy, not as "the accuracy coefficient"
- [ ] RQ1's tier null is cited from the cluster-robust OLS, not the M2 mixed model
- [ ] No non-converged coefficient is described as a null result
- [ ] The dip test reports p < 1e-16, never p = 0
- [ ] RQ3's mosque-finder gap is explained as speech-act difference, not harm
- [ ] 636 is never reported as qasr discussion volume (lexicon over-captures)
- [ ] RQ4's Δ is reported as M3's +0.085 (not −0.236), labelled not estimable,
      with the sign reversal under reweighting noted
- [ ] `tracker_score` at 1-of-1 broken is not presented as the most-broken feature
- [ ] "40 topics" never implies corpus coverage
- [ ] Reviewer names do not appear in any quote — anonymise before quoting
- [ ] No bare aspect count is reported for the seven aspects under 50% precision
      (§3.1a); each carries its precision or an adjusted interval
- [ ] RQ4's volume sentence does not present 2,194 as verified mentions
- [ ] Demand precision (§3.3) is either re-labelled on the current sheet, or
      reported with the stranded-sample caveat stated explicitly
