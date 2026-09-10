# Annotation Guide — Salah App Review Study

You are hand-labelling a sample of Google Play reviews so we can measure how
accurate our automated sentiment analysis is. The paper reports those accuracy
numbers, so **the quality of your labels is the quality of the paper's validity
claim**. Nothing checks your work afterwards — what you write *is* the ground
truth we measure the machine against.

Budget about **one hour per 300 rows**. Most reviews are very short: the median
review in this corpus is 4 words, and 56% are under 5 words.

**Contents**

1. [Who labels what](#1-who-labels-what)
2. [Getting the files](#2-getting-the-files)
3. [Opening them without corrupting them](#3-opening-them-without-corrupting-them)
4. [Sheet A — `doc_500` (review-level sentiment)](#4-sheet-a--doc_500-review-level-sentiment)
5. [Sheet B — `aspect_300` (aspect-level, lead only)](#5-sheet-b--aspect_300-aspect-level-lead-only)
6. [Sheet C — `demand_precision_sample` (lead only)](#6-sheet-c--demand_precision_sample-lead-only)
7. [Reviews you cannot read](#7-reviews-you-cannot-read)
8. [The shared rows — inter-rater reliability](#8-the-shared-rows--inter-rater-reliability)
9. [Finishing and returning the files](#9-finishing-and-returning-the-files)
10. [Quick reference card](#10-quick-reference-card)

---

## 1. Who labels what

| Who | File | Rows | Columns you fill |
|---|---|---|---|
| **Second annotator** | `doc_500_annotator2.csv` | 300 | `gold_label`, `annotator`, `notes` |
| Lead | `doc_500_annotator1.csv` | 300 | `gold_label`, `annotator`, `notes` |
| Lead | `aspect_300.csv` | 300 | `gold_aspect_correct`, `gold_aspect_true`, `gold_sentiment`, `gold_is_request` |
| Lead | `demand_precision_sample.csv` | 200 | `is_true_positive`, `correct_aspect`, `notes` |

`doc_500.csv` is the untouched 500-row reference copy. **Do not label it** — the
two `annotator` files are the split of it that people actually work on.

**Only touch the columns listed above.** Everything else — above all
`reviewId` — must stay exactly as it is. Labels are merged back by `reviewId`,
so if that column is altered the work cannot be recovered.

### Order of work

**Lead: do `aspect_300.csv` first.** It is the only sheet that blocks anything —
its labels set the zero-shot threshold, which gates re-running the aspect
extraction, which gates every remaining analysis stage. `doc_500` and
`demand_precision_sample` produce numbers for the Methods section and hold up
nothing downstream.

**Second annotator: start as soon as you get the file.** Krippendorff's α needs
both annotators' shared rows, so this is the other thing everything waits on.

Before either of you starts, spend ten minutes together agreeing the edge cases
in [section 4](#edge-cases-decided-in-advance). Ten minutes now prevents two
conscientious people drifting apart across 600 rows.

---

## 2. Getting the files

They live on the analysis server at
`~/Tonmoy/Salat-App-Research/src/data/gold_labels/`.

**Lead — pull them to your laptop:**

```bash
scp -r ubuntu@<server>:~/Tonmoy/Salat-App-Research/src/data/gold_labels ./gold_labels
```

**Second annotator:** you should receive exactly two things — this guide and
`doc_500_annotator2.csv`. You do not need the repository, the server, or any
other file. Do not ask for `doc_500_annotator1.csv`; seeing it would invalidate
the reliability measurement (see [section 8](#8-the-shared-rows--inter-rater-reliability)).

### English translations — already done

Roughly half of each `doc_500` file is not in English, so every non-English row
carries a **`content_en`** column with a machine translation. Use it. Its limits
are in [section 7](#7-reviews-you-cannot-read).

A small number of rows are still blank there — mostly gibberish or text Google
could not identify. That is expected; treat them as unreadable.

**Lead only — to top up the remaining blanks:**

```bash
uv pip install deep-translator
.venv/bin/python src/scripts/_translate_gold_rows.py --dry-run   # counts, no network
.venv/bin/python src/scripts/_translate_gold_rows.py --sleep 2.0
```

It only touches rows that are blank or previously failed; good translations are
left alone. Raise `--sleep` if the failure count climbs — Google throttles, and
below about 0.5 s it starts returning error pages instead of translations.

### Check the sheets before anyone starts

```bash
.venv/bin/python src/scripts/_verify_gold_sheets.py
```

This must print **`ALL CHECKS PASSED`** before labelling begins. It verifies row
counts, that the source text is undamaged, that no label column has been filled
by accident, that translations are real English rather than error pages or
untranslated echoes, and that the 100 shared rows are identical across both
annotator files with no leakage between the private halves.

> Why this gate exists: the first translation run silently wrote Google error
> pages (`Error 500 (Server Error)...`) into 412 of 546 rows and reported
> success. Nothing about the files looked wrong. Run the check.

---

## 3. Opening them without corrupting them

**Do not double-click the CSV.** About half the reviews are not in English —
Arabic, Bengali, Indonesian, Urdu, Malay and others. If the spreadsheet guesses
the wrong encoding, that text becomes garbage, and saving writes the damage back
permanently.

**LibreOffice Calc** — right-click the file → Open With → Calc. In the import dialog:

- Character set: **Unicode (UTF-8)**
- Separator options: **Comma** only (untick Tab, Semicolon, Space)
- Click the `reviewId` column heading, set Column type to **Text**
- String delimiter: `"`

**Excel** — open Excel first, then **Data → From Text/CSV**, choose the file:

- File Origin: **65001: Unicode (UTF-8)**
- Delimiter: **Comma**
- Click **Transform Data**, set `reviewId` to **Text**, then **Close & Load**

**Google Sheets** — File → Import → Upload → *Replace spreadsheet*. It handles
UTF-8 correctly and is the easiest option for sharing.

Save as **CSV UTF-8**, same filename.

### Verify the encoding before you start

Find any row where `lang` is `bn` or `ar` and look at `content_clean`:

| What you see | Meaning |
|---|---|
| Readable Bengali or Arabic script | Correct — carry on |
| `à¦¨à¦®à¦¾à¦œ`, `Ø§Ù„ØµÙ„Ø§Ø©`, or boxes | **Wrong encoding.** Close **without saving** and re-open per above |

If you have already saved a mangled file, do not try to repair it. Ask for a
fresh copy — the originals are committed in git and can be restored exactly.

---

## 4. Sheet A — `doc_500` (review-level sentiment)

Read the `content_clean` column. Decide what the reviewer thinks **about the
app, overall**. Write one of four values in `gold_label`, capitalised exactly:

| Label | Use when |
|---|---|
| `Positive` | Praises the app, raises no real complaint |
| `Negative` | Complains, offers no real praise |
| `Mixed` | Contains **both** genuine praise and a genuine complaint |
| `Neutral` | No evaluation at all — a question, a statement of fact, a greeting |

### Ignore the star rating

The `score` column is context only. **Do not let it decide your label.** A
central part of this study is reviews whose text disagrees with their stars — if
you copy the stars, that disagreement becomes invisible and the analysis loses
its point.

### Worked examples — review level

| Review | Label | Why |
|---|---|---|
| "Alhamdulillah, best app for prayer times. Never misses." | `Positive` | Praise only. Religious expression is not itself the sentiment — judge the appraisal of the app. |
| "Good app but the azan doesn't play on time since the update." | `Mixed` | Real praise **and** a real complaint. |
| "How do I change the calculation method to Hanafi?" | `Neutral` | A question. No evaluation. |
| "Wish it had a qibla widget." | `Neutral` | Asking for something absent is not a complaint about what exists. |
| "Too many ads. Uninstalling." | `Negative` | Complaint only. |
| "Masha Allah" | `Positive` | Brief approval is still approval. |
| "5 stars" | `Positive` | Explicit approval, even without detail. |
| "." / "aaaa" / a lone emoji | `Neutral` | No content to judge. |
| "Please fix the widget, it's been broken for months." | `Negative` | A request **about a feature that exists and is broken** is a complaint. Compare the qibla-widget row above, where the feature is simply absent. |
| "Was great until the last update." | `Negative` | Current judgement is negative; past praise does not make it Mixed. |
| "good" | `Positive` | Take short reviews at face value. |
| "ok" | `Neutral` | Bare acknowledgement, no clear approval. |

### Edge cases, decided in advance

- **Praise for Islam or gratitude to God, not the app** — "Alhamdulillah" alone,
  with nothing about the app: `Neutral`.
- **Spam, adverts, unrelated text**: `Neutral`, and note `spam` in `notes`.
- **A complaint about the phone, not the app** ("my Samsung kills the
  notification"): `Neutral`, note `not about app`.
- **Praise plus a feature request**: `Positive`. A request is not a complaint.
- **Praise plus a broken feature**: `Mixed`.
- **Only a star rating in the text** ("⭐⭐⭐⭐⭐"): `Neutral`, note `no text`.

### The rule people get wrong

**`Mixed` is a real category, not a place for reviews you are unsure about.**

Reviews that praise an app while condemning one specific thing are the entire
reason this study exists. Folding them into `Positive` or `Neutral` means the
number we publish stops measuring what we claim it measures.

If you are genuinely unsure, leave `gold_label` **blank** and write why in
`notes`. A blank is honest; a guess is noise.

---

## 5. Sheet B — `aspect_300` (aspect-level, lead only)

This validates the analytical core of the paper and gates the next pipeline
stage. Each row is **one sentence** plus what the model guessed about it.

Read `triggering_sentence`, then fill four columns:

| Column | Enter | Question you are answering |
|---|---|---|
| `gold_aspect_correct` | `1` / `0` | Is `predicted_aspect` what this sentence is actually about? |
| `gold_aspect_true` | aspect name | Only when you entered `0` — the correct aspect from the list below |
| `gold_sentiment` | `Positive` / `Negative` / `Neutral` / `Mixed` | Sentiment **toward that aspect**, not toward the whole review |
| `gold_is_request` | `1` / `0` | Is this asking for something **absent**, rather than judging something present? |

### When NO aspect applies

Some sentences carry no product content at all — pure praise, gratitude, or a
du'a: *"May Allah reward you"*, *"Assalamualaikum"*, *"thank you"*. The tagger
still fires on these because generic feature words collide with religious
register (`reward`, `guide`, `rank`).

For those rows: **`gold_aspect_correct` = 0 and leave `gold_aspect_true` blank.**

A blank `gold_aspect_true` therefore carries meaning — it says *"no aspect in
our taxonomy applies"*, as distinct from *"the right aspect is X"*. Both are
useful: the first measures how often the tagger fires on nothing, the second how
often it picks the wrong label. On the first 300 rows this split was 24% firing
on nothing and 28% picking the wrong aspect.

### Three aspects are narrower than they sound

These definitions are binding — the lexicon has been tightened to match them.

| Aspect | Counts | Does **not** count |
|---|---|---|
| `guides` | Salah and wudu guides only | Any other how-to or tutorial content |
| `calendar_sync` | Syncing with the user's own calendar so reminders adjust around times they are busy | The Hijri/Islamic calendar date display — a different feature entirely |
| `tracker_score` | Prayer trackers, their scores and streaks | Adhkar, Quran-reading or fasting trackers |


### The 27 aspects

The first 20 map to a feature column in the app spreadsheet; the last 7 are
review-only qualities that no app "has".

```
prayer_times_accuracy   madhab              calc_method         reminders_adhan
forbidden_times         nafl_times          goal_system         mosque_finder
guides                  adhkar              qibla               table_format
qasr_travel             calendar_sync       monetization        prayer_tracker
tracker_score           women_period        widgets             companion_hardware

ui_design               stability_bugs      ads_intrusive       privacy_data
quran_audio             complexity_bloat    spiritual_affect
```

### Worked examples — aspect level

| Sentence | correct | sentiment | request | Why |
|---|---|---|---|---|
| "Fajr is 12 minutes early in my city." | `1` | `Negative` | `0` | Predicted `prayer_times_accuracy`. The feature exists and is wrong. |
| "Please add auto qasr when I'm travelling." | `1` | `Neutral` | `1` | **The distinction the paper rests on.** Marking this Negative would invent dissatisfaction with a feature the app never had. |
| "The streak counter makes me feel guilty when I miss one." | `1` | `Negative` | `0` | RQ1's core phenomenon. |
| "Prayer tracking is my favourite part." | `0` → `prayer_tracker` | `Positive` | `0` | If the model predicted `privacy_data`, the word "tracking" misled it. |
| "Beautiful design but crashes every Ramadan." | `1` | `Positive` | `0` | If the predicted aspect is `ui_design`, sentiment **toward that aspect** is Positive. The crash belongs to `stability_bugs`. |
| "No Hanafi option anywhere in settings." | `1` | `Neutral` | `1` | Absence, not a judgement of a present feature. |

**`gold_sentiment` is about the aspect, not the review.** This is the single
most common mistake. A glowing review that criticises one feature yields
`Negative` for *that* aspect.

**Use `Mixed` here only** when the sentence is genuinely two-sided about the
*same* aspect — e.g. "the adhan is beautiful but far too quiet".

---

## 6. Sheet C — `demand_precision_sample` (lead only)

These 200 rows were matched by regular expressions, which are noisy. The paper
must report how noisy. Read `evidence_sentence` and answer one question:
**does this sentence really express the pattern named in `request_type`?**

| `request_type` | True positive when the sentence… |
|---|---|
| `request` | asks for a feature to be added or improved |
| `absence` | states that something is not there |
| `churn` | says they left, switched away, or uninstalled |
| `preference` | says why they chose this app over others |
| `tenure` | states how long they have used it |

Fill `is_true_positive` with `1` or `0`. If the pattern fired correctly but the
attached `aspect` is wrong, put the right aspect in `correct_aspect`.

| Sentence | Verdict | Why |
|---|---|---|
| "Missing the Hanafi asr option." | `1` | Matched `missing`; genuinely an absence/request. |
| "I wish everyone a blessed Ramadan." | `0` | Matched `wish`, but nothing is requested. The classic false positive. |
| "Switched to dark mode and it looks great." | `0` | Matched `switched to` as churn — but they switched a *setting*, not apps. |
| "Been using this since 2019." | `1` | Genuine tenure statement. |
| "Better than the alarm on my phone." | `0` | Matched `better than`, but the comparison is not to another app. |

> **Known issue this sheet resolves.** Our two independent request detectors —
> the ABSA `is_request` flag and these regex families — agree on only **55.6%**
> of request-type rows across the full corpus. This sheet is what turns that
> from an embarrassment into a reported measurement, so label it carefully.

---

## 7. Reviews you cannot read

Roughly **half of each `doc_500` file is not English** — check the `lang`
column. The sheet is stratified by language band on purpose, so non-English text
is deliberately over-represented relative to the corpus.

| File | Non-English rows | Main languages |
|---|---|---|
| `doc_500_annotator1.csv` | 140 of 300 (47%) | Arabic 62, Bengali 51, Urdu 5 |
| `doc_500_annotator2.csv` | 144 of 300 (48%) | Arabic 58, Indonesian 48, Bengali 23 |
| `aspect_300.csv` | 15 of 300 (5%) | mixed |
| `demand_precision_sample.csv` | 3 of 200 (2%) | mixed |

Note that this is **deliberately unrepresentative**: the corpus is 82% English,
but `doc_500` is stratified by language band so the non-English half gets
validated too. Do not be alarmed by how much of it you cannot read directly.

**Do not guess.** A label on text you cannot read is worse than no label,
because it adds noise to the very number meant to show our method is
trustworthy.

**Use the `content_en` column.** Every non-English row that could be translated
has one. Treat it as a rough guide, not gospel: machine translation flattens
sentiment, and the difference between `Positive` and `Mixed` often lives exactly
in the nuance it flattens. When a translation reads ambiguously, prefer a blank
over a guess.

These translations are a **reading aid only**. They never enter the analysis —
the pipeline scores non-English text natively with a multilingual model, for
precisely this reason.

**Where `content_en` is blank**, Google could not translate the row: usually
gibberish, keyboard mashing, or a script it failed to identify. If you cannot
read the original either:

1. Leave `gold_label` **blank**
2. Write `cannot read` in `notes`

Do not translate it yourself with another tool mid-task — that makes your rows
inconsistent with everyone else's.

We report precisely which languages the gold set covers. "Validated on English
and Bengali, n = X" is a normal, defensible thing for a paper to say. Silent
coin-flips on Arabic are not.

---

## 8. The shared rows — inter-rater reliability

Your file has an `is_overlap` column. The **100 rows where `is_overlap` is
TRUE** appear identically in the other annotator's file.

Those rows are the only source of Krippendorff's α — the statistic that tells
reviewers our labels are consistent *between people*, rather than one person's
private opinion. Without it, every qualitative claim in the paper is one
person's judgement.

**For those 100 rows:**

- Label them **independently**
- Do **not** discuss any of them with the other annotator while labelling
- Do **not** look at the other annotator's file
- Do **not** go back and change one to match after comparing

**Disagreement on these rows is expected and is itself the data.** Quietly
harmonising them does not improve the paper — it destroys the measurement and
makes the reported α meaningless.

If you both want to agree how to treat a *category* of review — one-word
reviews, spam, reviews about the phone rather than the app — settle it **before
you start**, and write the rule into [section 4](#4-sheet-a--doc_500-review-level-sentiment)
so it applies uniformly to everything.

> 39 of these 100 shared rows are Arabic. If neither annotator reads Arabic,
> translate first (section 2) or the α will rest largely on blanks.

---

## 9. Finishing and returning the files

1. Put your initials in the `annotator` column for every row you labelled
2. Use `notes` for anything odd — unreadable rows, spam, genuinely hard cases
3. Save as **CSV UTF-8**, same filename
4. Do not rename the file, reorder rows, or add/remove columns

**Second annotator:** send `doc_500_annotator2.csv` back to the lead.

**Lead — check the returned files before uploading:**

```bash
python - <<'PY'
import pandas as pd, glob
for f in sorted(glob.glob("gold_labels/*.csv")):
    d = pd.read_csv(f, dtype=str, keep_default_na=False)
    cols = [c for c in ("gold_label","gold_sentiment","is_true_positive") if c in d.columns]
    if not cols: continue
    filled = (d[cols[0]].str.strip() != "").sum()
    print(f"{f:48s} {filled:4d}/{len(d)} labelled")
PY
```

Then put them back on the server and commit, so the labels are versioned:

```bash
scp gold_labels/*.csv ubuntu@<server>:~/Tonmoy/Salat-App-Research/src/data/gold_labels/

# on the server
cd ~/Tonmoy/Salat-App-Research
git add src/data/gold_labels/*.csv
git commit -m "Gold labels: doc_500 complete"
git push
```

The gold sheets are deliberately **not** gitignored — they are hand-produced and
cannot be regenerated. Committing them after each session means a corrupted save
is always one `git checkout` away from recovery.

---

## 10. Quick reference card

```
doc_500_annotator*.csv
  gold_label  →  Positive | Negative | Neutral | Mixed   (blank if unreadable)
  annotator   →  your initials
  notes       →  anything unusual; "cannot read" for unreadable rows

aspect_300.csv
  gold_aspect_correct →  1 / 0
  gold_aspect_true    →  aspect name, only when the above is 0
  gold_sentiment      →  Positive | Negative | Neutral | Mixed
  gold_is_request     →  1 / 0

demand_precision_sample.csv
  is_true_positive →  1 / 0
  correct_aspect   →  aspect name, only when the aspect is wrong
```

- Ignore `score` when deciding sentiment
- `Mixed` = praise **and** complaint — never "unsure"
- A request for a **missing** feature is `Neutral` / `is_request = 1`
- A complaint about a **broken** feature is `Negative` / `is_request = 0`
- In `aspect_300`, sentiment is toward **the aspect**, not the review
- Blank beats a guess
- The 100 `is_overlap` rows are labelled blind, without discussion
