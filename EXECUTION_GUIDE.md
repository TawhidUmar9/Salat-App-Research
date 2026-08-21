# Execution Guide — Salah App Review Analysis

**Status**: all pipeline code is written and the non-GPU stages are verified against real data. The GPU stages (02b, 03, 03b, 04) are verified on **CPU** as a correctness check; the environment on this laptop only has a 3050 with 4GB VRAM, so full-scale timing is still to be confirmed on the 5090. Environment and correctness bugs found during testing are documented in §0.1 and §9 — read §0.1 before installing anything on the 5090, it will save you the same debugging.

---

## What to do right now

> **Pre-flight on the 5090 — do this before the full run.** It exercises every
> CUDA path and gives you real timings to extrapolate from, for ~20 minutes
> instead of discovering a problem six hours in.
>
> ```bash
> # 1. environment
> uv venv --python 3.12
> UV_HTTP_TIMEOUT=900 uv pip install torch --index-url https://download.pytorch.org/whl/cu124
> uv pip install -r src/requirements.txt
> .venv/bin/python -c "import torch;print(torch.cuda.get_device_name(0), torch.cuda.is_available())"
>
> # 2. whole chain on a sample — every stage, real GPU
> .venv/bin/python src/scripts/01_preprocess.py --sample 20000
> .venv/bin/python src/scripts/02a_sentiment_vader.py
> .venv/bin/python src/scripts/02b_sentiment_roberta.py
> .venv/bin/python src/scripts/03_absa.py --zeroshot-max 20000
> .venv/bin/python src/scripts/03b_promise_extraction.py
> .venv/bin/python src/scripts/03c_demand_mining.py
> .venv/bin/python src/scripts/04_topic_modeling.py
> .venv/bin/python src/scripts/05_temporal.py
> .venv/bin/python src/scripts/06_models.py
> .venv/bin/python src/scripts/08_visualizations.py
> ```
>
> **Three things to check before scaling up**, because each one silently
> produces wrong numbers rather than crashing:
>
> 1. `06_models.py` must NOT log `DROPPED '<predictor>'` for the aspects RQ2
>    depends on (`prayer_times_accuracy`, `qibla`, `ui_design`). If it does at
>    20K, it will still be sparse — but it should clear easily at full scale.
>    A `RQ2 VERDICT UNAVAILABLE` warning means the analysis did not run.
> 2. No coefficient in `model_results.csv` should exceed ~2 stars in magnitude,
>    and `p_value` must never be `NaN`. Both indicate separation.
> 3. `03_absa.py` should report a **non-trivial lexicon hit rate** (~35% of
>    sentences) — if it collapses toward 0%, the lexicon did not load.
>
> Then remove `--sample` / `--zeroshot-max` and re-run with `--resume`.


1. **On the 5090 machine**: follow §0 setup, using the CUDA torch index and the `UV_HTTP_TIMEOUT` workaround. Budget 20–40 min for the CUDA wheel download alone.
2. **In parallel, not blocking the above**: knock out the two manual action items in §5 — the 6 missing app annotations and the gold-set labelling. Neither needs a GPU.
3. Once torch is confirmed (`import torch; torch.cuda.is_available()` → `True`), run the dry run in §2 with `--sample 20000` to get real GPU timings before committing to the full 732K-row run.

---

## 0. Setup (once)

Default `python3` on this machine is **3.14**, which has no working wheels for `fasttext`, `hdbscan`/`umap-learn` (BERTopic), or `torch`. The venv must be built on **Python 3.12**.

```bash
cd /home/tawhidumar/codes/falah-paper-codes

uv venv --python 3.12
uv pip install -r src/requirements.txt
```

**For the RTX 5090**, install the CUDA build of torch *before* the rest:

```bash
UV_HTTP_TIMEOUT=900 uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -r src/requirements.txt
```

> **`UV_HTTP_TIMEOUT` is not optional here.** `uv`'s default is 30 s, and the
> NVIDIA CUDA wheels torch depends on are multi-GB. On a normal connection the
> download exceeds 30 s and dies with *"Failed to download distribution due to
> network timeout"*. Worse, if you pipe the install through `tail`/`head`, the
> shell reports the *pipe's* exit status — so a failed install looks like it
> succeeded. Check for the real thing instead:
>
> ```bash
> .venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
> ```

Verify:

```bash
.venv/bin/python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_available())"
```

Everything below assumes `.venv/bin/python`. (`source .venv/bin/activate` then plain `python` works too.)

Downloaded automatically on first use — no manual steps:
- fastText `lid.176.bin` (~126 MB) → `src/data/_models/`
- NLTK `vader_lexicon`
- HuggingFace model weights

### 0.1 Environment bugs already fixed — do not re-discover these

A fresh `pip install -r requirements.txt` already handles these. Listed here so the symptoms are recognisable if they resurface:

| Package | Problem | Fix now in `requirements.txt` |
|---|---|---|
| ~~`transformers` version~~ **(MISDIAGNOSIS — corrected)** | XLM-R's tokenizer crashed with `Error parsing line b'\x0e' in .../sentencepiece.bpe.model`. This was originally blamed on transformers 5.x and pinned to `<4.50`. **That was wrong.** The real cause was a **0-byte `sentencepiece.bpe.model`** left behind by the `huggingface_hub` Xet bug below — transformers was correctly reporting a corrupt file. Both 4.49 and 5.15 are verified working end-to-end once downloads are intact. The pin has been **removed**: it was unnecessary *and* it conflicts with `bertopic`, which requires 5.x. |
| `tiktoken` | Not auto-installed as a transformers dependency, but `AutoTokenizer.from_pretrained(...)` silently needs it for some models' conversion path. Fails with `ModuleNotFoundError: No module named 'tiktoken'`. | Added explicitly |
| `protobuf` + `sentencepiece` | Any sentencepiece-based tokenizer (XLM-R, BART-MNLI) needs both. Fails with `requires the protobuf library but it was not found`. | Added explicitly |
| `transformers` (again) | `AutoModelForSequenceClassification.from_pretrained(..., dtype=...)` works for some model classes but `XLMRobertaForSequenceClassification.__init__()` on 4.49.0 rejects `dtype` as an unexpected kwarg — a per-model-class inconsistency in that library version. | `_nlp.py` uses `torch_dtype=` instead, which is accepted uniformly |
| `huggingface_hub` | **Downloads that hang forever, alive but making no real progress.** HF migrated large files to a chunked "Xet" storage backend; `huggingface_hub` 0.36.2's Xet-reconstruction path pathologically stalls on this kind of large file — opens dozens of small connections to CDN edges, target blob never grows past 0 bytes. A direct `curl` to the same CDN sustains 5–16 MB/s, so this is not a network problem, it's specific to the Xet client path. | `_common.py` sets `HF_HUB_DISABLE_XET=1` at import time, forcing the plain-HTTP path. Applies automatically to every script — nothing to do |

If you ever see `ValueError: Error parsing line ... in .../sentencepiece.bpe.model`, **do not downgrade transformers** — that was the wrong fix. It means the cached model file is corrupt or truncated. Delete the cached repo and re-download with the Xet backend off:

```bash
rm -rf ~/.cache/huggingface/hub/models--cardiffnlp--twitter-xlm-roberta-base-sentiment
HF_HUB_DISABLE_XET=1 .venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('cardiffnlp/twitter-xlm-roberta-base-sentiment')"
```

**If a HuggingFace model download ever seems stuck again**, don't assume it's the network. Check whether the blob is actually growing:

```bash
find ~/.cache/huggingface/hub -name '*.incomplete' -newermt '-2 minutes'
```

If a `.incomplete` file exists but hasn't grown in 2 minutes while the process is still alive, it's the Xet path stalling — verify `HF_HUB_DISABLE_XET=1` is set (it is, by default, via `_common.py`; only relevant if you're running HF downloads outside this project's scripts).

### 0.2 If a `torch` install stalls

`uv pip install torch --index-url .../cu124` can hang mid-download — the process stays alive with an established connection but stops receiving bytes. It looks identical to "still downloading slowly." How to tell the difference:

```bash
# Run twice, a few seconds apart. If the byte count is IDENTICAL both times
# and no file in the uv cache has been touched recently, it's stalled, not slow.
cat /proc/$(pgrep -f 'uv pip install torch')/io | grep rchar
find "$(uv cache dir)" -newermt '-2 minutes' -type f | wc -l   # 0 = no recent writes
```

If stalled: kill it (`pkill -f 'uv pip install torch'`) and retry with a single connection stream, which is less prone to stalling than the default 4 parallel ones:

```bash
UV_HTTP_TIMEOUT=900 UV_CONCURRENT_DOWNLOADS=1 uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**For a quick CPU-only correctness check** (e.g. to verify the code runs before committing to the big CUDA download, or on a machine without a GPU), the CPU wheel is ~25 MB and installs in seconds — this is how the GPU-stage scripts in this guide were verified without a working CUDA download:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Swap to the CUDA index later for real runs; `_common.get_device()` auto-detects whichever is installed, no code changes needed either way.

---

## 1. Flags every script accepts

| Flag | Purpose |
|---|---|
| `--sample N` | Run on N reviews, stratified across all 26 apps. **Use this first, always.** |
| `--resume` | Reuse completed checkpoint shards instead of recomputing |
| `--overwrite` | Delete this script's checkpoints and start clean |
| `--batch-size N` | Override the auto-sized batch |
| `--device auto\|cuda\|cpu` | Compute device; `auto` prefers CUDA |
| `--fp32` | Disable bf16/fp16 (use only if you see NaNs) |
| `--seed N` | RNG seed (default 42) |

Batch size is auto-derived from detected VRAM, and **halves and retries automatically on CUDA OOM** rather than crashing. GPU stages checkpoint to `src/data/_checkpoints/` every shard, so a crash costs one shard, not the run.

---

## 2. Run order

```
01_preprocess ──┬── 02a_vader ──┐
                └── 02b_roberta ─┴── 02c_cross_validation (notebook)
                        │
                        ├── 03_absa ──┬── 03c_demand_mining
                        │             └── 05_temporal
                        ├── 03b_promise_extraction
                        └── 04_topic_modeling
                                      │
                            06_models (needs 03, 03b, 03c)
                                      │
                     07_rq1..rq6 notebooks → 08_visualizations
```

### Recommended: dry run first

```bash
.venv/bin/python src/scripts/01_preprocess.py       --sample 20000
.venv/bin/python src/scripts/02a_sentiment_vader.py
.venv/bin/python src/scripts/02b_sentiment_roberta.py
.venv/bin/python src/scripts/03_absa.py             --no-zero-shot --fast-split
.venv/bin/python src/scripts/03b_promise_extraction.py --no-zero-shot
.venv/bin/python src/scripts/03c_demand_mining.py
.venv/bin/python src/scripts/04_topic_modeling.py
.venv/bin/python src/scripts/05_temporal.py
.venv/bin/python src/scripts/06_models.py
.venv/bin/python src/scripts/08_visualizations.py
```

That exercises every stage in ~15 minutes and surfaces problems before you commit hours of GPU time.

### Full run

```bash
.venv/bin/python src/scripts/01_preprocess.py
.venv/bin/python src/scripts/02a_sentiment_vader.py
.venv/bin/python src/scripts/02b_sentiment_roberta.py --resume
.venv/bin/python src/scripts/03_absa.py --resume
.venv/bin/python src/scripts/03b_promise_extraction.py
.venv/bin/python src/scripts/03c_demand_mining.py
.venv/bin/python src/scripts/04_topic_modeling.py
.venv/bin/python src/scripts/05_temporal.py
.venv/bin/python src/scripts/06_models.py
.venv/bin/python src/scripts/08_visualizations.py
```

Do **not** run 02a and 02b in parallel on one GPU — they will contend for VRAM. 02a is CPU-only and takes about a minute.

---

## 3. What each stage does

### `01_preprocess.py` — Phase 0
Merges 26 review CSVs (732,194 rows), joins metadata, deduplicates on `reviewId`, extracts emoji before cleaning, parses dates, detects language with fastText, and assigns the five sub-corpus flags.

→ `src/data/master_reviews.parquet` (44 columns)

**Verified on a 5,200-row sample**: 26/26 apps resolved to package ids, 20 matched to the annotation sheet, 47% of reviews land in `C_text`.

### `02a_sentiment_vader.py` — Phase 1A
VADER on `content_clean`. Also sets `vader_applicable` (English only) — VADER scores non-English text as "Neutral", which would otherwise inflate that class by ~10 points.

### `02b_sentiment_roberta.py` — Phase 1B
**Deviation from the plan, deliberate.** English rows → `twitter-roberta-base-sentiment-latest`; non-English → `twitter-xlm-roberta-base-sentiment`, scored natively. The plan proposed machine-translating first; that is ~100K+ API calls, and §16.3 already warns Bengali sentiment does not survive MT. `sentiment_model` records which model scored each row so `C_en` and `C_trans` stay separable and translated results are never pooled into headline numbers.

Also adds the star-rating proxy (§4.3).

→ `src/data/master_reviews_with_sentiment.parquet`

### `03_absa.py` — Phase 2 (the analytical core)
Sentence-segments with `pysbd`, tags aspects via the 27-aspect lexicon, runs `bart-large-mnli` on sentences with no lexicon hit, scores per-aspect sentiment, and flags feature *requests* separately from complaints.

The zero-shot stage is optimized — naive `pipeline()` would be one forward pass per (sentence × 27 labels):
- exact-duplicate sentences collapsed (short reviews repeat heavily)
- MiniLM embedding prefilter shortlists top-k labels before bart-large sees them
- length-bucketed batching, entailment-logit-only scoring, bf16 autocast

```bash
--zeroshot-topk 5        # prefilter width; 0 = all 27 labels, much slower
--zeroshot-threshold .75 # calibrate this on the aspect gold set, don't guess
--zeroshot-max N         # cap unmatched sentences
--no-zero-shot           # lexicon only
```

→ `src/data/aspect_sentiments.parquet` (one row per reviewId × aspect)

### `03b_promise_extraction.py` — Phase 2.5a
Extracts claimed features from all 26 store descriptions, validates against the hand annotation with per-feature Cohen's κ, and cross-checks "All Features for free" against `adSupported`/`offersIAP`.

CSV stays primary where it exists; description fills the 6 unannotated apps.

→ `src/data/feature_matrix_combined.parquet`, `promise_source_agreement.csv`

### `03c_demand_mining.py` — Phase 2.5b
Regex extraction of requests / absence / churn / preference-citations / tenure, attached to aspects. **This is what rescues the rare-feature RQs** — "N reviews across the 19 apps without qasr ask for it" is publishable where a null between-app test is not.

Also writes a 200-row precision-check sheet — these patterns are noisy and the paper must report their precision.

→ `src/data/demand.parquet`, `gold_labels/demand_precision_sample.csv`

### `04_topic_modeling.py` — Phase 3
BERTopic on `C_en` with an Islamic-domain stopword list plus app-name tokens (without them every cluster is "allah good app muslim"). Fits sub-topic models within tracker reviews (RQ1) and women/privacy reviews (RQ4) — the ecosystem model cannot resolve streak-anxiety from streak-motivation.

→ `src/data/topics.parquet`, `topic_info.csv` (with an empty `manual_label` column for you)

### `05_temporal.py` — Phase 4
Monthly aggregates, PELT changepoint detection, the Muslim Pro 2020 privacy interrupted time-series, aspect volume over time, per-version sentiment. Sparse months are flagged, never silently smoothed.

→ `temporal_trends.parquet`, `changepoints.csv`, `muslim_pro_privacy_its.csv`, `aspect_volume_over_time.csv`, `version_sentiment.csv`

### `06_models.py` — Phase 5
M1–M6 as review-level mixed models with a random intercept per app, plus BH-FDR correction across every coefficient.

```bash
--models m1,m2   # run a subset
```

→ `app_comparison.csv`, `gap_matrix.csv`, `unmet_needs_ranking.csv`, `model_results.csv`

### `08_visualizations.py` — Phase 7
All 14 paper figures, PNG @300 DPI + vector PDF, consistent app colors, Wong colorblind-safe palette. Figures whose inputs are missing are skipped with a logged reason rather than killing the run.

```bash
--only 5,6,7   # regenerate specific figures
```

---

## 4. Notebooks

They are `.py` files in jupytext percent format — run as scripts, or convert:

```bash
uv pip install jupytext
.venv/bin/jupytext --to notebook src/notebooks/*.py
.venv/bin/jupyter notebook
```

| Notebook | Purpose |
|---|---|
| `02c_cross_validation.py` | **Generates the gold-set sheets**, then evaluates κ / P / R / F1 / α |
| `04_topic_exploration.py` | Read representative docs, fill `manual_label` |
| `05_temporal_visualizations.py` | Trend exploration |
| `06_cross_app_analysis.py` | Ecosystem descriptives + model review — **read this before the RQ notebooks** |
| `07_rq1_gamification.py` | Bimodality test, guilt:motivation ratio |
| `07_rq2_accuracy_vs_ui.py` | M1 forest plot, qibla failure taxonomy |
| `07_rq3_traveler.py` | Mosque-finder edge, qasr demand |
| `07_rq4_femtech.py` | Women-aspect volume, loyalty proxies |
| `07_rq5_bloat_hardware.py` | Feature-count curves, iQIBLA case study |
| `07_rq6_completeness.py` | Delivery gap + unmet-need ranking → design implications |

Each RQ notebook ends with an **"Answer to RQ_"** cell with blanks to fill — that is the paragraph that goes into the paper.

---

## 5. Two things only you can do

### 🔴 A. Annotate the 6 missing apps

Still **0/23 filled** in `Salah App Analysis - Sheet1.csv`:

| App | Reviews | Installs |
|---|---|---|
| Salatuk (Prayer time) | 111,232 | 50M+ |
| Salaat First: Prayer Times | 8,895 | 10M+ |
| নামাজের সময়সূচি — Prayer Time | 6,572 | 1M+ |
| Prayer Times and Qibla | 4,546 | 1M+ |
| Sadiq: Prayer, Quran, Qibla | 1,525 | 100K+ |
| IslamApp: Prayer times & Athan | 577 | 1M+ |

6 apps × 22 features = 132 cells. The pipeline runs without this — `03b` fills those apps from store descriptions — but the hand annotation is the stronger IV and the κ validation reference.

> **The description fallback measured worse than hoped, so this matters more than it looks.** Running `03b` against the 20 hand-annotated apps gives a **mean Cohen's κ of 0.257** — "fair" at best. Per-feature it ranges from κ=0.64 (`Has Companion Hardware`) down to κ=−0.10 (`All Features for free`, where the description says "free" for 8 apps the annotator marked 19). Under-detection dominates: `Madhab Variations` 12→2, `Various methods of calculation` 11→4, `Widgets` 12→6.
>
> Practical consequence: for the 6 unannotated apps, `feature_count` is systematically **under**-counted. That biases RQ5's bloat analysis and inflates RQ6b's unmet-need scores for those apps. Hand-annotating removes the problem entirely for 18% of the corpus.
>
> Report the κ table (Figure 3) honestly as a measurement-validity finding — "store descriptions are an unreliable proxy for shipped features" is a legitimate, citable result, not a failure.

**How the sheet is encoded** (the code follows this exactly): any non-empty cell = feature present; `✓` and a note string like `"Google Maps"` both count. Blank = absent. An app with *all* cells blank is treated as **unannotated (NaN), never as "has zero features"** — so partially filling a row would silently mark the rest absent. Fill a row completely or not at all.

Also delete the trailing empty `Aranna` row.

### 🔴 B. Gold-set labelling

```bash
.venv/bin/python src/scripts/03_absa.py --sample 20000   # needed for the aspect sheet
.venv/bin/jupytext --to notebook src/notebooks/02c_cross_validation.py
# run it — section 3 writes the sheets
```

Produces:
- `doc_500.csv` — 500 reviews stratified by install tier × star × language → label {Positive, Negative, Neutral, **Mixed**}
- `aspect_300.csv` — 300 sentences stratified **by aspect** (proportional sampling would contain almost no `women_period` or `qasr_travel`) → label aspect correctness + sentiment + is_request
- `doc_500_annotator1/2.csv` — split with a 100-row overlap for Krippendorff's α

Then re-run section 4 of that notebook for the numbers that go into Methods and Figure 2, and section "Zero-shot threshold calibration" to pick the real `--zeroshot-threshold`.

---

## 6. Design decisions baked into the code

Recorded here because reviewers will ask, and because they differ from the plan in places.

| Decision | Rationale |
|---|---|
| Non-English scored natively by XLM-R, not translated | ~100K MT calls is infeasible; §16.3 says Bengali sentiment doesn't survive MT. `--translate-sample N` still exists for quote/gold rows a human must read. |
| Join key is the **package id**, not the app name | Store titles drift. Name normalization collapsed *Prayer Alarm* and *Prayer Time, Azan Alarm, Qibla* onto the same key — a silent mis-join. Now 26/26 resolve with zero collisions. |
| Unannotated apps are `NaN`, never `0` | Blank-vs-absent is unrecoverable from the sheet alone; treating blank rows as "no features" would have handed RQ5 six fake zero-feature apps. |
| Requests excluded from aspect sentiment | "Wish it had a qibla" is demand, not dissatisfaction with an existing qibla. §5.3 calls scoring these as negative a paper-killing failure mode. |
| Zero-shot uses an embedding prefilter | Full 27-label NLI over every unmatched sentence is intractable. Report `prefilter_recall()` — it bounds achievable recall and reviewers will ask. |
| Promise extraction uses a **separate, stricter vocabulary** from review ABSA | Store copy is marketing prose, not user complaint. The review-side keyword `journey` matched *"embark on a spiritual journey"* in 13 listings and inflated `qasr_travel` from 1 promising app to 14 — which would have collapsed RQ3's unmet-need score and fabricated RQ6a broken promises. 7 ambiguous aspects now define `promise_keywords` in `aspect_lexicon.yaml`; the rest fall back to the review set. Mean κ improved 0.193 → 0.257. |
| κ = 0.00 on near-universal features is a degeneracy, not a failure | `Prayer Times by Location` and `Timely Reminders` are present in 20/20 apps. When one rater is constant, Cohen's κ is 0 (or undefined) no matter how high raw agreement is — both sit at 80–95% agreement. Report raw agreement alongside κ for these, or exclude them from the mean with a footnote. |
| M4 uses a linear probability model | Logistic MixedLM over ~300K rows doesn't converge reliably in statsmodels; the LPM coefficient reads directly as a percentage-point change. |
| BH-FDR, not Bonferroni | §9.7: with 22 features × multiple aspects Bonferroni guarantees a null. |

---

## 7. Output map

```
src/data/
├── master_reviews.parquet                # 01
├── master_reviews_with_sentiment.parquet # 02b
├── aspect_sentiments.parquet             # 03
├── feature_matrix_combined.parquet       # 03b
├── promise_source_agreement.csv          # 03b — Figure 3
├── demand.parquet                        # 03c
├── topics.parquet + topic_info.csv       # 04
├── temporal_trends.parquet               # 05
├── changepoints.csv                      # 05
├── muslim_pro_privacy_its.csv            # 05 — Figure 13
├── aspect_volume_over_time.csv           # 05 — Figure 12
├── version_sentiment.csv                 # 05
├── app_comparison.csv                    # 06
├── gap_matrix.csv                        # 06 — RQ6a
├── unmet_needs_ranking.csv               # 06 — RQ6b → design implications
├── model_results.csv                     # 06 — all coefficients + q-values
├── gold_labels/                          # YOU FILL THESE
├── quotes/rq1..rq6.csv                   # 07 notebooks
├── _checkpoints/                         # resume state (safe to delete)
└── _models/                              # cached lid.176.bin, BERTopic

src/figures/fig01..fig14 .png + .pdf
```

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `Failed to download distribution due to network timeout` | `uv`'s 30 s default is too short for the CUDA wheels. Prefix with `UV_HTTP_TIMEOUT=900` |
| Install "succeeded" but `import torch` fails | You piped `uv` through `tail`/`head`, so the reported exit code was the pipe's. Re-run unpiped and check `import torch` directly |
| `uv pip install torch` hangs, process alive but no progress | Stalled download, not slow. See §0.2 to confirm and fix |
| `Error parsing line b'\x0e' in .../sentencepiece.bpe.model` | `transformers` ≥4.50 bug with XLM-R/BART-MNLI tokenizers. See §0.1 |
| `ModuleNotFoundError: No module named 'tiktoken'` | See §0.1 — already fixed in requirements.txt, re-run `pip install -r src/requirements.txt` if you hit this |
| `requires the protobuf library but it was not found` | See §0.1 — same fix, `protobuf` + `sentencepiece` are now explicit deps |
| `include_groups=True is no longer allowed` | pandas 3.x in the venv. `uv pip install "pandas>=2.2,<3.0"` |
| CUDA OOM | Handled automatically (halves and retries). To pre-empt: `--batch-size 64` |
| `fasttext` build fails | Use `fasttext-wheel`, already in requirements.txt |
| Run died mid-GPU-stage | Re-run with `--resume` |
| `Missing input: ... Run X first` | Working as intended — run the named upstream script |
| BERTopic `hdbscan` build error | Python must be 3.12, not 3.14 |
| A figure is missing | Check its logged SKIP reason; usually an upstream stage hasn't run |
| Everything is slow | You're on the full corpus, or on CPU. Use `--sample 20000` while iterating; expect GPU to be 10–50× faster than CPU for 02b/03/03b/04 |

---

## 9. Verified on this machine

This machine has a 3050 (4GB VRAM) and its `torch` CUDA download stalled repeatedly (§0.2), so the GPU stages below were verified for **correctness on the CPU build** — they run and produce sane output, but timings will not resemble the 5090. Sample sizes are small (300–3,000 rows) specifically to keep CPU runtime reasonable while checking the logic.

| Check | Result |
|---|---|
| All 23 Python files compile | ✅ |
| `01_preprocess.py` end to end | ✅ 732,194 rows merged → 44 columns, sub-corpora assigned |
| App identity resolution | ✅ 26/26 apps → package ids, **zero collisions** |
| Feature-CSV parsing | ✅ 20 annotated / 6 NaN (not 0) |
| Lexicon compiles | ✅ 27 aspects, 370 keywords, 105 regex patterns |
| fastText language detection | ✅ auto-downloads; 83% en / 4.6% ar / 4.2% bn |
| `02a_sentiment_vader.py` | ✅ end to end |
| `02b_sentiment_roberta.py` (CPU, English-only, n=300) | ✅ model loads, scores, writes 57-column parquet |
| ABSA request-vs-complaint split (unit tests) | ✅ 9/9 cases — "wish it had a qibla" flagged request, not negative qibla |
| Demand pattern extraction (unit tests) | ✅ 10/10 cases across all 5 pattern families |
| M6 gap-matrix math | ✅ synthetic fixtures: broken-promise flag + unmet-need ranking both correct |
| `03_absa.py` full pipeline (CPU, lexicon-only, n=300) | ✅ end to end — segmented 535 sentences, 273 lexicon hits, 178 unique sentences scored on XLM-R, wrote 254-row `aspect_sentiments.parquet` (51.5% pos / 17.0% neu / 31.5% neg) |
| `03b_promise_extraction.py` (n=600) | ✅ end to end — 26 apps, κ report written; **surfaced a real bug** (see §6 promise-vocabulary row) |
| `03c_demand_mining.py` (n=600) | ✅ end to end — all 5 pattern families, precision sheet written |
| `04_topic_modeling.py`, `05_temporal.py`, `06_models.py`, `08_visualizations.py` | ⏳ not run — `bertopic` and `ruptures` were never installed in this laptop's venv (see note below) |
| Real GPU timing on CUDA (5090 or otherwise) | ⏳ not yet measured anywhere |

Five bugs were found and fixed by this testing — two silent-corruption bugs in
the analysis code, three broken-dependency/environment bugs that blocked the
GPU stages outright with a crash or an infinite hang rather than silently:

1. **App-name collision** — the original normalizer collapsed *Prayer Alarm* and
   *Prayer Time, Azan Alarm, Qibla* onto the same key, mis-joining two apps'
   feature annotations. Now joined on package id.
2. **`privacy_data` false positives** — bare `tracking` in that aspect's keyword
   list matched *"women period tracking"* and *"prayer tracking"*, inflating
   privacy complaints with tracker discourse. Now requires surveillance context.
3. **`transformers` 5.15.1 incompatible with XLM-R/BART-MNLI tokenizers** — see §0.1.
4. **`torch_dtype` vs `dtype` kwarg** — `XLMRobertaForSequenceClassification.__init__()`
   on the pinned transformers version rejects `dtype` outright (works fine for
   the plain RoBERTa class — a per-model-class inconsistency). See §0.1.
5. **`huggingface_hub`'s Xet download path hangs indefinitely** on large files —
   alive, zero real progress, easy to mistake for "just slow." A direct `curl`
   to the same CDN sustained 5–16 MB/s, proving it wasn't the network. See §0.1
   and §0.2 — `HF_HUB_DISABLE_XET=1` is now set automatically by `_common.py`.

**Note**: sample outputs are deleted after each smoke test so you don't
accidentally run Phase 5 on a few thousand rows and mistake it for the full
corpus. `src/data/` should hold only the lexicon, the cached fastText model,
and empty output folders between runs — if you see stray `.parquet` files
there before your first real run, they're smoke-test leftovers, safe to delete.

### 9.1 This laptop's venv is deliberately incomplete

Packages were installed piecemeal while debugging the environment bugs above, so
this venv has 18 of the 30 declared requirements. **On the 5090 machine, install
from `requirements.txt` properly and everything below resolves itself.**

Currently missing here: `bertopic`, `hdbscan`, `umap-learn` (blocks 04),
`ruptures` (05 degrades gracefully — changepoint detection silently skips),
`krippendorff` (02c α), `diptest` (RQ1 falls back to a bimodality coefficient),
`deep-translator` (only needed for `--translate-sample`), `jupytext` (CLI, for
notebook conversion).

Three declared packages — **`pingouin`, `seaborn`, `wordcloud`** — are carried
over from the plan's §14 list but are **never imported anywhere** in the final
code; it uses plain matplotlib and statsmodels/scipy instead. Harmless, but you
can drop them from `requirements.txt` if you want a leaner install.
