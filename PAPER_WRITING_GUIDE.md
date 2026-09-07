# How to write this paper

Three documents work together. [`PAPER_DISCLOSURES.md`](PAPER_DISCLOSURES.md) is
what must be said and what may not be claimed. [`RESULTS_DIGEST.md`](RESULTS_DIGEST.md)
is every number with its caveat. This is the plan: what the paper argues, what
goes in each section, and the order to write it in.

The six **"Answer to RQ_"** cells in `src/notebooks/07_rq*.py` are already written
in full sentences. They are drafted prose, not notes — most of the results section
is assembling them, not composing from scratch.

---

## 1. The argument

Write toward one claim, not six.

> **In devotional apps, satisfaction is governed by the reliability of core
> functions rather than by feature richness — and three standard HCI framings
> mis-locate the problem.**

Each RQ retires a framing and puts something better in its place:

| Framing the literature offers | What the data shows |
|---|---|
| Gamification induces streak anxiety | Guilt is *rarer* in tracker discourse (OR 0.28; 3 of 77 codes). The complaint is an untrustworthy record and a broken prompt |
| Users trade accuracy against aesthetics | The classes cost the same, but prayer-time accuracy alone costs 1.85× interface — and 41.6% of accuracy complaints sit inside 4–5 star reviews |
| Differentiating features win users | Users cite adhan, Quran audio and accuracy. Mosque finder is not in the top eight |
| Feature bloat drives complaints | Feature count is uncorrelated; the leanest apps draw bloat complaints at 3.3× the rate of the fullest. Bloat is an interface property (23.98× co-occurrence) |
| Apps under-serve niche needs | Two distinct failures: coverage (market) and delivery (quality) — and delivery fails on the *most basic* promises |

**Three threads recur; name them explicitly rather than letting a reader find them.**

1. **Advertising.** Costliest complaint in the corpus (−1.439), a dominant
   `rq5_bloat` topic, and `Monetization against religious purpose` in the RQ1
   coding. Spans RQ1, RQ2, RQ5, RQ6.
2. **Adhan reminders.** Most-cited reason for choosing an app (374 preference
   citations), joint most-broken promise (5 of 20 apps), and the dominant tracker
   failure theme at 46%. Spans RQ1, RQ3, RQ6.
3. **The record, not the streak.** RQ1's reframe, RQ4's menstrual exemption,
   RQ6's coverage gap — users want an accurate, owned, accommodating record of
   their worship.

---

## 2. Contributions to claim

State four. Two empirical, two methodological — the methods pair is what lifts
this above a review-mining paper.

1. **An ecosystem-scale account of devotional app quality** — 732,194 reviews,
   26 apps, six pre-specified questions, with the FDR denominator reported in full.
2. **A reframing of gamification harm** in religious practice: not guilt, but
   record integrity, prompting reliability, and the app mis-modelling worship.
3. **Devotional register breaks off-the-shelf NLP.** `goal_system` scored 0%
   precision because reviewers write "May Allah reward you"; 1,686 formulaic
   sentences required filtering. This is a finding about applying general-purpose
   sentiment and aspect tooling to religious text, not a cleaning footnote.
4. **Two sampling hazards in app-store research**, both demonstrated with
   quantities:
   - *Star ratings are an unsafe proxy for defect severity* — 41.6% of accuracy
     complaints arrive inside 4–5 star reviews.
   - *Single-app qualitative sampling manufactures genre-level themes* — themes
     A, B and F were 51% of the single-app pass and 4% of the cross-app pass.

---

## 3. Section plan

### Abstract
Corpus size, the six questions in one clause, then the three reframings and the
two methods hazards. Lead with the counterintuitive: guilt is rarer, not commoner;
lean apps draw more bloat complaints, not fewer.

### Introduction
Prayer apps are used daily by a very large population and are studied mostly as
instances of habit tracking. Set up the three framings from §1, promise to test
them at ecosystem scale, and state the four contributions.

### Related work
Four strands: religious/devotional technology in HCI; gamification and
self-tracking (this is where streak anxiety lives — cite it fairly and fully,
because you are contradicting it); app-store review mining as method; feature
bloat and interface complexity. Signal the reframing in the gamification
paragraph so the results don't ambush the reader.

### Method
Follow the pipeline order: corpus and scraping → language filtering → sentiment →
aspect lexicon → **devotional-formula filtering** (contribution 3, give it real
space) → demand mining → topic modelling → models. Then measurement quality: α =
0.824, precision figures with the 192/300 explanation, the discarded zero-shot
backstop.

**Everything in [`PAPER_DISCLOSURES.md`](PAPER_DISCLOSURES.md) §2–3 belongs here.**
Report the non-converged coefficients openly (§4.1) — disclosing 11 unusable
terms in a stated denominator of 29 reads as rigour, and a reviewer who finds them
unaided reads it as concealment.

### Results
One subsection per RQ, in order. Start from the answer cells. Each needs:
its headline claim, the numbers with intervals, the claim ceiling from the
disclosures, and one or two anonymised quotes.

**Where a null appears, say which model produced it.** RQ1's tier null and RQ5's
feature-count null both come from the cluster-robust OLS because the mixed models
did not converge. Write that.

### Discussion
Three moves.
- **The reframings**, and what each implies for design. Reliability and
  accommodation, not gentler streak mechanics.
- **The two gaps** from RQ6: coverage is a market failure, delivery a quality
  failure on the most basic promises. Different problems, different remedies.
- **Methods implications** — contribution 4, addressed to anyone doing app-store
  research, not only to this domain.

### Limitations
Sampling ceilings (§1.1), topic-model coverage (§2.1), the RQ4 sub-model collapse
(§2.3), the `qasr_travel` over-capture (§4.4), N=1 hardware and N=3 inclusion,
and the provenance split for RQ5's framing (§5). Written straight, no hedging
language — each has a claim ceiling already stated.

### Design implications
Drive from RQ6b's unmet-need ranking, stated in ecosystem terms rather than
per-app. Companion hardware, menstrual accommodation, travel concession, calendar
integration. Pair with the delivery point: fix adhan reliability and monetization
before adding anything.

---

## 4. Order of writing

Not front to back. Write what is most settled first, so the argument stabilises
before the framing is committed.

1. **Method** — fully determined, nothing to decide
2. **Results**, RQ1 → RQ6, assembled from the answer cells
3. **Limitations** — lift from the disclosures
4. **Discussion** — write only after the results are on the page; the argument
   will have shifted
5. **Introduction and related work** — last, so they promise exactly what is
   delivered
6. **Abstract** — very last

---

## 5. Rules that hold everywhere

- **Never pool the two coding passes.** Report side by side; the divergence is
  the finding.
- **"Not estimable" ≠ "no effect."** Eleven coefficients cannot support
  inference. RQ4 and the hardware case study are descriptive only.
- **Cite nulls to the model that converged**, naming it.
- **Anonymise every quote.** `export_quotes` never writes `userName`, but check
  the text itself for self-identification before it reaches the paper.
- **The app is not the subject.** Name applications only where the finding is
  genuinely app-specific (themes A and B, the Muslim Pro privacy episode).
- **RQ5's test was pre-specified; its framing was not.** Say both.

---

## 6. Before submission

Run the checklist in [`PAPER_DISCLOSURES.md`](PAPER_DISCLOSURES.md) §8 twice —
once when the draft is complete, once on the final PDF. Then check the
"figures that must never appear" table in
[`RESULTS_DIGEST.md`](RESULTS_DIGEST.md) against the finished text.

Ship the server's `run_manifest.jsonl` with the artefact submission.
