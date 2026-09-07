# Disclosures — what the paper must say out loud

Every item here is something a reader could otherwise be misled about. Each one
has a **claim ceiling** (what the evidence will actually support) and, where it
helps, a sentence you can adapt. Work through it once while drafting and once
before submission.

Two provenance markers are used throughout:

- ✅ **verified** — checked directly against the files in this repo on 2026-09-07
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

Six of ten topics in `rq4_women_privacy` carry the same theme after labelling
(data selling to the US military), and a seventh is a near-duplicate — 63% of the
sub-model's documents under one label. The Muslim Pro data-selling story swamps
the subset.

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

### 3.2 Two aspects have no gold rows 📋

`calendar_sync` and `goal_system` (0.2% of corpus volume) are excluded from the
weighted figure.

### 3.3 Demand-pattern precision is uneven 📋

**91.5%** overall, with `absence` weakest at **65%**. Report the weak one.

### 3.4 Promise-source agreement is low 📋

Mean κ = **0.272**. This is weak agreement and must be stated as such wherever
promise data is used.

### 3.5 The zero-shot backstop was dropped 📋

Measured **11.5%** precision on 300 hand-labelled sentences, so it was removed.
Reporting a discarded component is evidence of calibration, not weakness.

### 3.6 Devotional formulae were filtered — this is a contribution 📋

1,686 sentences (0.8%) removed before aspect tagging. Generic English feature
vocabulary misfires badly on religious register: `goal_system` scored **0%**
precision because reviewers write "May Allah reward you". Frame this as a finding
about applying off-the-shelf NLP to devotional text, not as a cleaning step.

### 3.7 Inter-rater agreement 📋

Krippendorff's α = **0.824**.

---

## 4. Statistics

### 4.1 Eleven coefficients cannot support inference, not three ✅

**Verified against `model_results.csv` from the 2026-09-07 run.** "Not estimable"
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

**Star ratings hide accuracy complaints** ✅ (computed 2026-09-07). Of 7,942
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

### 4.4 RQ3 — check the direction before writing 📋

Mosque finder β = −0.137 (p = 0.003) from the cluster-robust fit. Apps that have
the feature get more negative mentions of it, which may simply be because you can
only complain about something that exists. The mixed model gave +0.05 with a NaN
standard error, so it is not trustworthy — but a sign flip deserves a sentence.

### 4.5 RQ4 — not estimable, descriptive only 📋

2,194 mentions across 24 apps, 272 demand signals, but only **3 apps** have the
feature. Say plainly that no between-app estimate is possible. Combine with the
sub-model collapse in §2.3.

### 4.6 RQ5 — lead with the fused cases, report the bound 📋

Co-occurrence lift **23.98×** (OR 29.41, p = 7.4e-137) across all co-occurring
reviews; **7.62×** (OR 8.78, p = 9.3e-24) restricted to separate sentences. Of
134 reviews carrying both complaints, 94 (70%) fuse them in one sentence and 40
(30%) voice them separately. The two lexicons share no keyword and no regex, so a
fused sentence contains two distinct terms — it is not one phrase counted twice.
Report 7.62× as the conservative bound.

✅ The labelled `rq5_bloat` topics support this from a different direction, and
note *what* they contain: they are dominated by simplicity as something users
**praise** (`App praised for simplicity` three times, `Praise for minimalistic
UI`, `App praised for having no ads`) rather than by feature counting. Bloat talk
is interface talk.

### 4.7 RQ6 📋

18 broken promises; unmet needs led by companion hardware, women's tracking and
mosque finder.

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
| Krippendorff's α | 0.824 |
| Aspect precision, weighted | 84.4% [79.7, 88.9] |
| Coefficients surviving q<0.05 | 20 of 29 |
| Not estimable | 3 |

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
- [ ] "40 topics" never implies corpus coverage
- [ ] Reviewer names do not appear in any quote — anonymise before quoting
