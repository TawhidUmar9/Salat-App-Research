# Manual Work Guide — What's Left, and How to Do It

The pipeline is finished. Every number exists. What remains is the human work
that turns those numbers into a paper: naming the topics, reading the quotes,
and writing the answers.

This guide is for **both of you**. Section 1 has two decisions we need to make
together before the writing starts. Sections 2–4 are the tasks, in order.

---

## 1. Two questions to answer first

### Q1 — How do we frame RQ5?

The pre-registered test came back positive, twice over:

| | lift | odds ratio | p |
|---|---|---|---|
| All co-occurring reviews | **23.98×** | 29.41 | 7.4e-137 |
| Separate sentences only | **7.62×** | 8.78 | 9.3e-24 |

Of the 134 reviews carrying both a bloat complaint and an interface complaint,
**94 (70%) fuse them into one sentence** — *"cluttered UI"*, *"confusing and not
intuitive"* — and 40 (30%) voice them separately.

The two lexicons share no keyword and no regex, so a fused sentence contains two
distinct terms. It is not one phrase being counted twice.

**Option A — Conservative.** Report only the 7.62× figure from separate
sentences. Modest, entirely free of any dependence on how sentences were split.

**Option B — Substantive (recommended).** Lead with the fused cases: users
describe bloat *as a property of the interface*, not as a count of features.
Report 7.62× as the conservative bound. This is the sharper HCI claim and the
`rq5_bloat` topics support it independently.

> **Our answer:**
> _(write A or B here, with the date)_

Decide before reading any more results, and write it down. That is what makes it
a pre-specification rather than a story fitted afterwards.

### Q2 — Do we want a results digest?

Every headline figure, its confidence interval, its q-value, and the caveat each
one needs — on a single page, so nobody re-derives them from terminal scrollback
at 2am. Includes the exact sentences for the precision, stability, and
co-occurrence caveats.

> **Our answer:**
> _(yes / no)_

---

## 2. Getting the files

**As of 2026-09-06 all five sheets are already in `src/data/` and committed.**
Skip to section 3 and start labelling. The rest of this section is how they got
there, and how to pull them again on another machine.

The sheets are pipeline output, and `.gitignore` excludes `src/data/*.csv`.
They live on the analysis server, same as the gold sheets:

```bash
scp '<user>@<server>:~/Tonmoy/Salat-App-Research/src/data/topic_info*.csv' src/data/
scp '<user>@<server>:~/Tonmoy/Salat-App-Research/src/data/quotes/rq1_tracker_negative_50.csv' src/data/quotes/
```

Keep the single quotes — the `*` must expand on the server.

The three sub-model sheets carry `representative_doc_1` and `_2`, so they can be
labelled in a spreadsheet with nothing else installed. Only the 40-topic
`topic_info.csv` really wants the notebook, which shows 15 documents per topic
instead of 2. For that you also need the parquets:

```bash
scp '<user>@<server>:~/Tonmoy/Salat-App-Research/src/data/topics.parquet' src/data/
scp '<user>@<server>:~/Tonmoy/Salat-App-Research/src/data/master_reviews_with_sentiment.parquet' src/data/
```

Labels are hand-made and not reproducible. Once a sheet is filled, copy it back
to the server and commit it, the way the gold labels were handled.

---

## 3. Label the topics — the big one

**Why this matters more than it sounds.** BERTopic gives you keyword salads like
`simple, ads, big, read, translation`. A reader cannot use that. Your label is
what turns a cluster into a finding, and every RQ notebook downstream prints
your labels, not the keywords.

### How to do it

```bash
.venv/bin/jupytext --to notebook src/notebooks/04_topic_exploration.py
.venv/bin/jupyter notebook          # or open it in VSCode
```

For each topic the notebook shows the top keywords and a handful of
**representative documents**. Read the documents — not just the keywords — then
write a short label into the `manual_label` column of the CSV.

Four files need labelling:

| File | Topics | Who |
|---|---|---|
| `topic_info.csv` | 40 | split between you |
| `topic_info_rq5_bloat.csv` | 10 | **lead — highest priority** |
| `topic_info_rq4_women_privacy.csv` | 10 | either |
| `topic_info_rq1_tracker.csv` | 3 | either, quick |

### What makes a good label

| Keywords | ✅ Good label | ❌ Bad label |
|---|---|---|
| `azaan, notification, sound, hear, working, doesn` | Adhan fails to play | Notifications |
| `premium, subscription, paid, restore, account` | Subscription and restore-purchase problems | Premium |
| `widget, widgets, size, font, bigger, screen, small` | Widget text too small to read | Widgets |
| `simple, easy, simple easy, useful` | Praise for simplicity | Simple |
| `hai, ki, ka, bhi, aap, se, ke` | Hindi/Urdu — not resolvable | Language |

Rules:
- **Name the complaint or the praise, not the noun.** "Widgets" is a topic area;
  "Widget text too small to read" is a finding.
- **Six words or fewer.**
- If a topic is incoherent, label it `MIXED — unusable` and say so. Do not force
  a name onto noise.
- Topic `-1` is the outlier bucket. Label it `OUTLIERS` and move on.

### Two things about the 40-topic sheet

**Topic 0 is a 168,054-document catch-all** — 61% of everything assigned, with
keywords `useful, helpful, ads, amazing, read, bless, easy`. It is generic
praise and nothing finer. Label it `MIXED — generic praise` and do not build a
claim on it. The other 39 topics are properly differentiated. This is ordinary
BERTopic behaviour on a review corpus reduced to 40 topics, not a defect.

**22.7% of documents are outliers** (62,745 of 276,422 assigned). Report that
share in Methods alongside the ARI figures.

### Topics you can label quickly

From the last run, these read clearly from their keywords alone — confirm
against the representative docs, but they should be fast:

- `azaan, notification, sound, hear, working` → adhan playback failure
- `open, update, working, crashing, fix, keeps` → crashes after update
- `premium, subscription, paid, restore` → subscription problems
- `date, calendar, hijri, calender, wrong` → Hijri date wrong
- `direction, qiblat, qiblah, wrong` → qibla direction wrong
- `location, gps, manually, city, detect` → location detection
- `widget, size, font, bigger, small` → widget legibility
- `ui, interface, minimalist, minimal, clean` → praise for minimal interface
- `slow, phone, fix, open, bit slow` → app is slow
- `ads, unnecessary, annoying, remove` → intrusive advertising

### ⚠️ Two honest caveats to carry into the write-up

**`rq5_bloat` is only moderately reproducible.** Across five random seeds it
produced 11, 10, 7, 11 and 6 topics, mean pairwise ARI **0.452**. The *themes*
recur every time — simplicity, widget size, slowness, unnecessary ads,
minimalism — but the exact partition moves.

So: **claim the themes, not the partition.** Write "bloat complaints cluster
around legibility, speed and advertising" rather than "we identified exactly ten
bloat topics". Report the ARI and the 6–11 range in Methods.

**`rq1_tracker` gave only 3 topics** (ARI 0.837, so that structure is stable —
there just is not much there). One is Hindi/Urdu stopwords, one is privacy, and
one is a single large `track, log, daily` cluster. That is too thin to carry
RQ1. Lean on section 4 instead.

---

## 4. Code the 50 tracker-negative reviews — RQ1's real evidence

`src/data/quotes/rq1_tracker_negative_50.csv`

RQ1's statistics came back **null**: no difference in sentiment between apps with
a tracker, a tracker-plus-score, or neither (p = 0.49, p = 0.35). But two other
things are true and interesting:

- Tracker sentiment is **bimodal** (Hartigan's dip D = 0.118) — people love it or
  hate it, not a spread in between.
- Guilt language is **under-represented** in tracker reviews, not over —
  guilt:motivation runs 0.26 versus a corpus baseline of 0.86 (Fisher OR = 0.28,
  p = 3.5e-11). **The opposite of the streak-anxiety hypothesis.**

So the question these 50 reviews answer is: *when people do dislike the tracker,
what is actually wrong?*

### Protocol

Add a **`theme_code`** column and give each review a short phrase. Then group
the phrases into themes. The column must be named `theme_code` — that is what
`07_rq1_gamification.py` reads.

> ⚠️ **Do not re-run `07_rq1_gamification.py` after you start coding.** Its
> section 6 rewrites this file with an empty `theme_code` column. Keep a copy
> outside `src/data/` while you work. Do not start from a fixed list — let the categories come from the
text, then consolidate.

Watch for:
- **Data loss** — "lost my whole history after the update"
- **Accuracy** — the tracker marking prayers wrongly
- **Guilt or pressure** — if it appears at all; the statistics say it is rarer
  than the literature would predict, which is itself the finding
- **Friction** — too many taps to log a prayer
- **Theological discomfort** — unease at quantifying worship

If guilt turns out to be rare here too, that is a real result and belongs in the
paper. A null against a well-known hypothesis, backed by 6,285 reviews, is worth
more than a weak confirmation.

Aim for one sitting. 50 short reviews is about an hour.

---

## 5. Run the RQ notebooks

Only after topic labels are filled — the notebooks print your labels.

```bash
.venv/bin/jupytext --to notebook src/notebooks/06_cross_app_analysis.py
# then 07_rq1.py ... 07_rq6.py
```

Read `06_cross_app_analysis` first: it is the ecosystem overview and gives you
the context the individual RQs assume.

Each `07_rq*` notebook ends in an **"Answer to RQ_"** cell. That cell becomes a
paragraph of the paper. Write it in full sentences, not notes.

### What each answer has to say

**RQ1 — gamification.** Null on tier. Bimodal sentiment. Guilt under-represented
against the hypothesis. Carry the qualitative codes from section 4.

**RQ2 — accuracy vs interface.** Accuracy −0.647 stars, interface −0.601.
**Statistically indistinguishable.** Do not claim accuracy wins; the honest
finding is that they cost about the same. Note that `madhab` flips sign in the
ordinal check (β = +0.22, p = 0.65).

**RQ3 — competitive edge.** Mosque finder β = −0.137 (p = 0.003) from the
cluster-robust fit. **Check the direction before writing it up:** apps that have
the feature get more negative mentions of it, which may simply be because you can
only complain about something that exists. Also note the mixed model gave +0.05
and the clustered OLS gave −0.14; the mixed model had a NaN standard error, so it
is not trustworthy, but a sign flip deserves a sentence.

**RQ4 — inclusion.** 2,194 mentions across 24 apps, 272 demand signals, but only
**3 apps** have the feature. Not estimable — descriptive only. Say so plainly.

**RQ5 — bloat.** Per your answer to Q1 above.

**RQ6 — gaps.** 18 broken promises; unmet needs led by companion hardware,
women's tracking and mosque finder.

---

## 6. Numbers you will need

| Measure | Value |
|---|---|
| Corpus | 732,194 reviews, 26 apps |
| Analysed (has text) | 324,687 |
| Topic-modelled (C_en) | 276,422 in 40 topics |
| Outlier bucket (topic −1) | 62,745 (**22.7%**) |
| Inter-rater agreement (Krippendorff's α) | **0.824** |
| Aspect precision, corpus-weighted | **84.4%** [79.7, 88.9] |
| Aspect precision, unweighted | 69.3% |
| Demand-pattern precision | 91.5% (`absence` weakest at 65%) |
| Promise-source agreement (mean κ) | 0.272 |
| Coefficients surviving BH-FDR q<0.05 | 20 of 29 |
| Not estimable (report as such, never as null) | 3 |

### Caveats that must appear in Methods

1. **Aspect precision is estimated on 192 of 300 gold rows** — those the current
   configuration still produces. The gold set was labelled against an earlier,
   broader lexicon; every change since only removes matches, so the surviving
   rows are an unbiased subset rather than a re-labelling.
2. **`calendar_sync` and `goal_system` have no gold rows** (0.2% of corpus
   volume) and are excluded from the weighted figure.
3. **Topic structure varies by seed.** Report the ARI and count range for every
   sub-model, especially `rq5_bloat` at 0.452.
4. **Devotional formulae were filtered** before aspect tagging — 1,686 sentences
   (0.8%). Generic English feature vocabulary misfires badly on religious
   register (`goal_system` scored 0% precision because reviewers write "May Allah
   reward you"). This is a contribution, not just a cleaning step.
5. **The zero-shot backstop was dropped** after measuring 11.5% precision on 300
   hand-labelled sentences.
6. **Three models are not estimable** through separation and are named in the
   output. Never report them as null results.

---

## 7. Order of work

~~Pull the sheets~~ and ~~regenerate `topic_info.csv`~~ — both done 2026-09-06.
All five sheets are in `src/data/` and committed. Start at step 1.

| # | Task | Who | Rough time |
|---|---|---|---|
| 1 | Answer **Q1 and Q2** in section 1 | together | 10 min |
| 2 | Label `topic_info_rq5_bloat.csv` (10 topics) | **lead** | 45 min |
| 3 | Label `topic_info_rq4_women_privacy.csv` (10) + `topic_info_rq1_tracker.csv` (3) | partner | 1 hr |
| 4 | Label `topic_info.csv` — 40 topics, split 20/20 | both | 1½ hr each |
| 5 | Code the 50 tracker-negative reviews (`theme_code` column) | **lead** | 1 hr |
| 6 | Commit and push the sheets (section 8) | either | 5 min |
| 7 | Run `06_cross_app_analysis`, then `07_rq1`…`07_rq6` | either | — |
| 8 | Write | together | — |

Steps 2–5 run in parallel between the two of you. Step 7 needs every label in
place **and pushed**. Q1 in step 1 must be answered before anyone reads further
results — that is what makes it a pre-specification.

### Why `topic_info.csv` was regenerated (done — 2026-09-06)

Runs before 2026-09-06 finished with the ecosystem sheet overwritten: every
sub-model in `04_topic_modeling.py` wrote through the same hardcoded path, so
`topic_info.csv` ended up holding the last sub-model (`rq5_bloat`) instead of
the 40 ecosystem topics. The bug is fixed in `label_topics`, and
`_regen_topic_info.py` rebuilt the sheet from the saved BERTopic model and
`topics.parquet` — no refit. The sheet now in the repo is the recovered one:
40 topics, verified not to be a duplicate of the bloat sheet.

Only that one file was ever affected. The sub-model sheets were always correct,
and no statistics depend on it.

If it ever needs rebuilding again:

```bash
# on the server
.venv/bin/python src/scripts/_regen_topic_info.py
```

Representative documents in a regenerated sheet are picked by topic probability,
not BERTopic's own c-TF-IDF ranking — the saved model does not serialize
`representative_docs_`. Deterministic, and the notebook shows fifteen per topic
against these two anyway.

---

## 8. When the labelling is done

Commit the sheets. They are tracked now, so the diff shows exactly the labels
you added and nothing else:

```bash
git add src/data/topic_info*.csv src/data/quotes/rq1_tracker_negative_50.csv
git commit -m "Label the topic sheets and code the tracker-negative reviews"
git push
```

**Push before running any RQ notebook on the server** — it reads the sheets from
the repo, so unpushed labels mean the notebooks print empty ones.
