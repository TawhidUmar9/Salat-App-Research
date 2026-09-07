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

### 1.1 The 50 tracker codes come from one app ✅

All 50 coded reviews are from *Athan: Prayer Times & Al Quran*. This is an
artefact of the selection rule, not a property of the corpus: `06_models.py`
takes `nsmallest(50, "sent_num")`, `sent_num` is three-valued (−1/0/+1), so every
negative review ties at −1 and pandas returns the first fifty in frame order —
which is app order.

**Claim ceiling:** a single-app case study. Not evidence about the ecosystem.

> The qualitative codes were drawn from the fifty most negative tracker reviews
> by model sentiment. Because sentiment is discretised, the selection resolved
> ties in corpus order and returned reviews from a single application; we
> therefore treat these codes as an illustrative case study of that application
> rather than a cross-app sample.

**If you run the extension draw**, replace the above with the stratified
description and report the app spread.

### 1.2 "Negative" means sentiment, not stars ✅

18 of the 50 carry 4–5 star ratings. Selection is by RoBERTa sentiment on the
review text, so the set includes otherwise-positive reviews containing one
specific complaint.

**Write:** "reviews expressing tracker dissatisfaction" — not "negative reviews".

### 1.3 Codes must be consolidated before they are reported ✅

There are 33 distinct codes across 50 reviews. The raw 33 are not a result;
report the consolidated themes and say how many codes collapsed into them.

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

### 4.1 Never report the not-estimable models as null 📋

Three models fail through separation and are named in the output. "Not
estimable" and "no effect" are different claims. 20 of 29 coefficients survive
BH-FDR at q < 0.05; report the full denominator.

### 4.2 RQ1 — the null is the finding 📋

No difference in sentiment by tracker tier (p = 0.49, p = 0.35). Sentiment is
bimodal (Hartigan's D = 0.118). Guilt language is **under-represented**:
guilt:motivation 0.26 against a corpus baseline of 0.86, Fisher OR = 0.28,
p = 3.5e-11 — the *opposite* of the streak-anxiety hypothesis.

✅ The qualitative codes corroborate this independently: **guilt or judgment
appears in 2 of 50** coded reviews, while **logging integrity dominates at
roughly 19 of 50** (redesign breaking the log, backfill windows, lost history,
counts regressing). Two independent methods, same direction — say so; it is the
strongest thing in the paper.

Subject to the single-app ceiling in §1.1.

### 4.3 RQ2 — the two costs are indistinguishable 📋

Accuracy −0.647 stars, interface −0.601. **Do not claim accuracy wins.** The
honest finding is that they cost about the same. Note that `madhab` flips sign in
the ordinal check (β = +0.22, p = 0.65).

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
- [ ] The tracker codes are described at their single-app ceiling (or the
      extension draw is done and described instead)
- [ ] "40 topics" never implies corpus coverage
- [ ] Reviewer names do not appear in any quote — anonymise before quoting
