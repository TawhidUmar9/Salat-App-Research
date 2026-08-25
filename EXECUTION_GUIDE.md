# Execution Guide — Salah App Review Analysis

**Status**: all pipeline code is written and verified against real data. Stages `01`–`03c` are now **confirmed working on the 5090** with real GPU timings (§9); `04`–`08` at full scale are still outstanding. Two things to know before you run anything: torch must come from the **`cu128`** index, not `cu124` (§0.1 — `cu124` has no `sm_120` kernels and fails *after* reporting `cuda.is_available() == True`), and `03b` must be run with **`--no-zero-shot`** (§5A — the NLI backstop inflates `feature_count` by 44% on the description-filled apps and corrupts RQ5's IV). Other environment and correctness bugs are documented in §0.1 and §9.0.

---

## What to do right now

**Done:** environment on the 5090 (`cu128`), the 20K pre-flight across every stage, and the hand annotation — the sheet is 26/26 (§5A). **Next:** the full corpus, then the gold-set labelling.

### Step 1 — Full corpus, stages 01→03c  ·  ~30 min, unattended

```bash
.venv/bin/python src/scripts/01_preprocess.py
.venv/bin/python src/scripts/02a_sentiment_vader.py
.venv/bin/python src/scripts/02b_sentiment_roberta.py --resume
.venv/bin/python src/scripts/03_absa.py            --resume
.venv/bin/python src/scripts/03b_promise_extraction.py --no-zero-shot
.venv/bin/python src/scripts/03c_demand_mining.py
```

Confirm in the log before moving on:

| Expect | Meaning if wrong |
|---|---|
| `Feature CSV encoding: … explicit-absent ('x') …` and `26 apps — 26 annotated` | The `x` parsing regressed — `feature_count` is corrupt (§5A) |
| `Resolved to packages: 26/26` | An app failed to join; feature data is mis-assigned |
| `Lexicon hits: … (≈35%)` | Near 0% means the lexicon did not load |
| No `DROPPED` for `prayer_times_accuracy` / `qibla` / `ui_design` | RQ2 is not estimable |

### Step 2 — Generate the labelling sheets  ·  minutes

```bash
uv pip install jupytext jupyter
.venv/bin/jupytext --to notebook src/notebooks/02c_cross_validation.py
.venv/bin/jupyter notebook    # run sections 1–3
```

### Step 3 — Label by hand  ·  the long pole, days

`doc_500.csv` (+ the two annotator splits), `aspect_300.csv`, `demand_precision_sample.csv`. See §5B for what each column means. Nothing else can proceed past step 4 without `aspect_300`.

### Step 4 — Calibrate the threshold, then re-run 03  ·  ~20 min

Run the notebook's "Zero-shot threshold calibration" section against the filled `aspect_300.csv`, then:

```bash
.venv/bin/python src/scripts/03_absa.py --zeroshot-threshold <calibrated>
```

`--resume` is safe here: the checkpoint fingerprints the threshold and clears stale shards by itself if it changed (§0.3). Leaving it off costs one extra NLI pass, nothing more.

### Step 5 — Final analysis and figures  ·  04 is the slow one

```bash
.venv/bin/python src/scripts/03c_demand_mining.py
.venv/bin/python src/scripts/04_topic_modeling.py
.venv/bin/python src/scripts/05_temporal.py
.venv/bin/python src/scripts/06_models.py
.venv/bin/python src/scripts/08_visualizations.py
```

Re-run `03c` first — its input changed when the threshold did. `04` is CPU-bound (UMAP/HDBSCAN, no GPU) and is the one stage whose full-scale cost is unmeasured.

### Step 6 — Notebooks → paper

`06_cross_app_analysis.py` first, then `07_rq1..rq6`. Each ends in an "Answer to RQ_" cell that becomes a paragraph. Fill `manual_label` in `topic_info.csv` via `04_topic_exploration.py` before the RQ notebooks lean on topic names.

---

## 0. Setup (once)

Default `python3` on this machine is **3.14**, which has no working wheels for `fasttext`, `hdbscan`/`umap-learn` (BERTopic), or `torch`. The venv must be built on **Python 3.12**.

```bash
cd /home/tawhidumar/codes/falah-paper-codes

uv venv --python 3.12
uv pip install -r src/requirements.txt
```

**For the RTX 5090**, install the CUDA build of torch *before* the rest — and use the **`cu128`** index, not `cu124` (see §0.1, "sm_120 not compatible"):

```bash
UV_HTTP_TIMEOUT=900 uv pip install torch --index-url https://download.pytorch.org/whl/cu128
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

> `torch.cuda.is_available() == True` only means a CUDA device is visible — it
> does **not** mean the installed build has kernels for that device's compute
> capability. On the 5090 (`sm_120`), a `cu124` build reports `True` here and
> then crashes on the first real op. Confirm a kernel actually runs:
>
> ```bash
> .venv/bin/python -c "import torch; x=torch.randn(2000,2000,device='cuda'); torch.cuda.synchronize(); print('OK', (x@x).sum().item())"
> ```
>
> If that raises `CUDA error: no kernel image is available for execution on
> the device`, see §0.1 "sm_120 not compatible".

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

> **`cu124` torch build is not compatible with the RTX 5090 — confirmed on the real 5090 machine.** Installing per the old §0 instructions (`--index-url .../cu124`) succeeds and `torch.cuda.is_available()` returns `True`, but with this warning:
>
> ```text
> UserWarning: NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible
> with the current PyTorch installation. The current PyTorch install supports CUDA
> capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
> ```
>
> The 5090 is Blackwell (`sm_120`); the `cu124` wheels only ship kernels up to `sm_90`. `is_available()` just checks that a CUDA device is visible — it does not check kernel compatibility, so this warning is easy to miss and the failure surfaces later, mid-pipeline, as a hard crash on the first real tensor op (`CUDA error: no kernel image is available for execution on the device`) inside `02b`/`03`/`03b`/`04`. Fix: use the **`cu128`** index instead, which has `sm_120` kernels.
>
> **`uv venv` does not include a `pip` module** — `.venv/bin/python -m pip uninstall ...` fails with `No module named pip` (silently, if you don't check the exit code) rather than removing anything. Worse, `uv pip install torch --index-url ...cu128` *without first removing the old build* does not fix it either: `uv` sees a package named `torch` already installed, reports `Checked 1 package` and does nothing, regardless of which index it originally came from. Use `uv pip uninstall`, not `pip`, and force the reinstall:
>
> ```bash
> uv pip uninstall torch torchvision torchaudio
> UV_HTTP_TIMEOUT=900 uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu128
> .venv/bin/python -c "import torch; x=torch.randn(2000,2000,device='cuda'); torch.cuda.synchronize(); print('OK', (x@x).sum().item())"
> ```
>
> If `cu128` ever turns out not to carry `sm_120` kernels either (it does as of early 2026), fall back to the nightly index: `--index-url https://download.pytorch.org/whl/nightly/cu128`. §0 and the pre-flight block above have already been updated to use `cu128`.

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

`uv pip install torch --index-url .../cu128` can hang mid-download — the process stays alive with an established connection but stops receiving bytes. It looks identical to "still downloading slowly." How to tell the difference:

```bash
# Run twice, a few seconds apart. If the byte count is IDENTICAL both times
# and no file in the uv cache has been touched recently, it's stalled, not slow.
cat /proc/$(pgrep -f 'uv pip install torch')/io | grep rchar
find "$(uv cache dir)" -newermt '-2 minutes' -type f | wc -l   # 0 = no recent writes
```

If stalled: kill it (`pkill -f 'uv pip install torch'`) and retry with a single connection stream, which is less prone to stalling than the default 4 parallel ones:

```bash
UV_HTTP_TIMEOUT=900 UV_CONCURRENT_DOWNLOADS=1 uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

**For a quick CPU-only correctness check** (e.g. to verify the code runs before committing to the big CUDA download, or on a machine without a GPU), the CPU wheel is ~25 MB and installs in seconds — this is how the GPU-stage scripts in this guide were verified without a working CUDA download:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Swap to the CUDA index later for real runs; `_common.get_device()` auto-detects whichever is installed, no code changes needed either way.

### 0.3 `--resume` invalidates itself when settings change

Checkpoint shards are keyed by **position** in the deduplicated work list, not by content. So changing anything that alters *which* items get computed — `--zeroshot-threshold`, `--zeroshot-topk`, `--zeroshot-max`, `--no-zero-shot`, `--sample` — makes cached shard *N* describe different rows than a freshly computed shard *N* would. Resuming across that boundary used to merge the two by sentence text, leaving every newly-included sentence with a **`NaN` sentiment that silently dropped out** of the results.

`Checkpoint` now writes a `_config.json` alongside the shards and re-checks it on `--resume`. If the fingerprint differs it names the changed setting and clears the stale shards automatically:

```text
WARN  Checkpoint 'absa': settings changed since the cached run
      (zeroshot_threshold) — discarding stale shards.
WARN      zeroshot_threshold: 0.75 -> 0.62
```

That warning is expected and correct after calibration (step 4) — it means the run is recomputing rather than reusing results from the old threshold. `--resume` is therefore always safe to leave on; it degrades to a full recompute exactly when it must.

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
.venv/bin/python src/scripts/03b_promise_extraction.py --no-zero-shot
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

### ✅ A. Annotate the 6 missing apps — DONE

`Salah App Analysis - Sheet1.csv` is now **26/26 annotated, 0 unannotated**. The description fallback is no longer load-bearing for any app; `03b`'s κ table becomes a pure measurement-validity result rather than a dependency.

The completed sheet changed the schema in three ways the code has been updated to match:

| Change | Effect |
|---|---|
| Absence is now an explicit **`x`**, not a blank (302 `x`, 322 present, **0 blanks**) | `load_feature_csv()` reads `x` as absent via `ABSENT_MARKERS`. **This was a silent-corruption bug**: the old parser treated *any* non-empty cell as "present", so every `x` would have counted as a feature. Every app would have scored 24/24 and `feature_count` — RQ5's independent variable — would have been a constant with zero variance, with no error raised |
| Two columns renamed — `Useful Adhkars` → `Salah Specific Adhkars`, `Goal System and Other events` → `… (streak)` | `FEATURE_COLUMNS` and `ASPECT_TO_FEATURE` updated; the `adhkar` and `goal_system` aspect mappings follow the new names |
| Two features added — `Life Qaza Calculator`, `Habit builder and tracker` | `FEATURE_COLUMNS` is now 24 (20 aspect-mapped + 4 unmapped). Unmapped features count toward `feature_count` but take no part in `03b`'s promise matching |

Resulting `feature_count`: range **7–16**, mean 12.4, sd 2.3, 9 distinct values across 26 apps — genuine spread for M4. Package resolution stays 26/26 with zero collisions.

The rare-feature constraints that shape RQ3/RQ4/RQ5 are unchanged, so those research questions keep their existing designs: `Auto Qasr mode` N=1, `Has Companion Hardware` N=1, `Women tracking` N=3. Two features are now present in **all 26** apps (`Prayer Times by Location`, `Timely Reminders`) — expect the documented κ=0 degeneracy on those (see §6), and report raw agreement instead.

> **Everything downstream must be recomputed.** `feature_count` changed for all 26 apps, not just the 6: the column set grew from 22 to 24, and `x` cells that the old parser would have counted as *present* are now correctly counted as *absent*. Any `feature_count`, `gap_matrix`, or `unmet_needs_ranking` produced before this point is stale. Re-run from `01_preprocess.py`.

> **The κ table is now a finding, not a dependency.** With 26/26 hand-annotated, no app's `feature_count` comes from a store description any more — `03b`'s agreement numbers exist purely to answer "how good a proxy would store copy have been?", and the answer is *poor*. Against the 20 apps annotated under the old schema it gave **mean Cohen's κ = 0.257**, ranging from κ=0.64 (`Has Companion Hardware`) to κ=−0.10 (`All Features for free`, where the description claims "free" for 8 apps the annotator marked 19). Under-detection dominated: `Madhab Variations` 12→2, `Various methods of calculation` 11→4, `Widgets` 12→6.
>
> **Those numbers are stale** — they predate the completed sheet, the renamed columns, and the two added features. Recompute them from the next `03b --no-zero-shot` run before citing anything in Figure 3.
>
> Report it honestly as a measurement-validity result: *"store descriptions are an unreliable proxy for shipped features"* is legitimate and citable, and it is now a cleaner claim than before, because the comparison is against a complete hand annotation rather than a partial one.

#### 🔴 Always run `03b` with `--no-zero-shot`

**The NLI backstop makes the description proxy strictly worse, and flips the bias from under- to over-counting.** Measured on the 5090 at 20K sample — the descriptions themselves don't depend on `--sample`, so these numbers are the real ones:

| | lexicon only (`--no-zero-shot`) | + zero-shot (default) |
|---|---|---|
| Mean Cohen's κ | **0.257** | 0.202 |
| `Various methods of calculation` (csv=11) | 4 — under | **14** — over, κ=−0.15 |
| `Has Companion Hardware` (csv=1) | κ=**0.64** | 5, κ=0.27 |
| `Connect to Google Calendar` (csv=1) | — | **16**, κ=0.03 |
| `Useful Adhkars` (csv=6) | — | **18**, κ=0.09 |
| `Salah and/or Wudu Guides` (csv=4) | — | **14**, κ=0.19 |

`ZS_THRESHOLD` is already 0.85 and the code comment at `03b_promise_extraction.py:46` anticipates exactly this — store copy is written to sound like it offers everything, so NLI entailment against *"This app offers {}."* is near-vacuous for feature labels.

**Why this corrupts RQ5 specifically.** With zero-shot on, the three highest `feature_count` apps in the whole corpus are all description-derived (Sadiq 16, iMuslim 16, Salatuk 14), and 4 of the top 8. Mean `feature_count` is **13.0 across the 6 description-filled apps vs 9.0 across the 20 hand-annotated** — a 44% inflation perfectly correlated with annotation source. `feature_count` is RQ5's independent variable, so that is a source artifact masquerading as a bloat finding. Lexicon-only leaves a smaller bias in the opposite direction; hand-annotating the 6 apps (above) removes it outright.

**How the sheet is encoded** (the code follows this exactly): any non-empty cell = feature present; `✓` and a note string like `"Google Maps"` both count. Blank = absent. An app with *all* cells blank is treated as **unannotated (NaN), never as "has zero features"** — so partially filling a row would silently mark the rest absent. Fill a row completely or not at all.

Also delete the trailing empty `Aranna` row.

### 🔴 B. Gold-set labelling

> **Run the full corpus BEFORE generating these sheets.** Every sheet is a
> *sample drawn from whatever is currently in `src/data/`*. Generate them from a
> `--sample 20000` run and you are labelling a sample of a sample — and
> `aspect_300` in particular gets a far thinner pool for exactly the rare aspects
> it exists to cover (`qasr_travel` has ~35 candidate sentences at 20K vs ~1,300
> at full scale). The GPU stages take ~25 min (§9); the labelling takes days.
> Order accordingly.

```bash
# 1. full corpus first — models are cached, this is ~25 min unattended
.venv/bin/python src/scripts/01_preprocess.py
.venv/bin/python src/scripts/02a_sentiment_vader.py
.venv/bin/python src/scripts/02b_sentiment_roberta.py --resume
.venv/bin/python src/scripts/03_absa.py --resume
.venv/bin/python src/scripts/03c_demand_mining.py

# 2. then generate the sheets from full-corpus data
uv pip install jupytext jupyter
.venv/bin/jupytext --to notebook src/notebooks/02c_cross_validation.py
.venv/bin/jupyter notebook    # run sections 1–3
```

The notebook cannot be run as a plain script — it calls Jupyter's `display()`. For `aspect_300.csv` only, without Jupyter: `.venv/bin/python src/scripts/_gen_aspect_gold_standalone.py`.

**Four things need hand-labelling, not two:**

| Sheet | n | You fill | Feeds |
|---|---|---|---|
| `gold_labels/doc_500.csv` | 500 | `gold_label` ∈ {Positive, Negative, Neutral, **Mixed**} | Methods κ/P/R/F1, Figure 2 |
| `gold_labels/doc_500_annotator1/2.csv` | 300 each, 100 shared | same | Krippendorff's α |
| `gold_labels/aspect_300.csv` | 300 | `gold_aspect_correct`, `gold_sentiment`, `gold_is_request` | ABSA validation **+ the zero-shot threshold** |
| `gold_labels/demand_precision_sample.csv` | 200 | `is_true_positive` (1/0) | §6.2 per-family precision — the regex patterns are noisy and the paper must report this |

`doc_500` is stratified by install tier × star × language; `aspect_300` is stratified **by aspect** on purpose (proportional sampling would contain almost no `women_period` or `qasr_travel`); `demand_precision_sample` is stratified across the five pattern families.

**`aspect_300` gates a second run of `03`.** The zero-shot threshold must be *calibrated* on it, not guessed — so the sequence is: run `03` at the default 0.75 → label `aspect_300` → run the notebook's "Zero-shot threshold calibration" section → **re-run `03_absa.py --zeroshot-threshold <calibrated>`** → then `03b` onward. Budget for `03` running twice; at ~17 min each that is cheap.

Then run section 4 of the notebook for the numbers that go into Methods and Figure 2.

> **Not a labelling sheet:** `quotes/rq1_tracker_negative_50.csv` is written by `06_models.py` — 50 tracker-negative reviews as raw material for RQ1's qualitative coding, not a validation set.

**Your labels are backed up, not clobbered.** These sheets are regenerated by re-running their producing stage, so `_common.write_gold_sheet()` checks for filled ACTION columns first and copies the old file to `<stem>.backup-<timestamp>.csv` before overwriting, logging a `WARN`. It never blocks the write. If you see that warning, the regenerated sheet is a **new sample** — merge on `reviewId`, never by row position.

---

## 5.5 RQ5 reframe — committed 2026-08-26, BEFORE the full-corpus run

**This is a pre-specification. Its whole value is the date.** The 20K pre-flight showed `feature_count` does *not* predict bloat complaints, and if anything points the other way (β=−0.0056 on `stability_bugs`, p=0.0056; app-level ρ=+0.44 sentiment, ρ=+0.49 rating, with `log_installs` controlled). The framing below was fixed at that point and must not be revised after seeing full-corpus numbers — revising it then is HARKing, and it is the first thing a CHI reviewer probes when a null becomes a finding.

**Old RQ5:** *do apps with more features attract more bloat complaints?* → answered **no**.

**Reframed RQ5:** *bloat complaints are real but are not about feature count — they are about how features are surfaced.*

The pipeline now produces both halves of that argument:

| Evidence | Where | Status |
|---|---|---|
| `feature_count` → bloat complaints | M4 MixedLM + cluster-robust OLS sensitivity | **confirmatory**, expected null — report the null with both fits |
| bloat × `ui_design` co-occurrence | `_bloat_surfacing()`, Fisher exact on one pre-specified 2×2 | **confirmatory**, one test, enters the BH-FDR family |
| lift of the other 25 aspects within bloat reviews | same function | **exploratory**, logged and deliberately NOT tested |
| what bloat complaints actually say | `04` sub-topic model `rq5_bloat` → `topic_info_rq5_bloat.csv` | qualitative, fitted on `complexity_bloat` only |

`ui_design` is the single confirmatory pair, chosen in advance. The sub-topic model is fitted on `complexity_bloat` **alone**, deliberately not pooled with `ui_design` — pooling would build the surfacing conclusion into the input instead of testing for it.

**If the co-occurrence test comes back null**, the honest paper says so: bloat complaints are unrelated to both feature count *and* interface complaints, and the reframe failed. Write that sentence now so the temptation to fish later is gone.

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
| `UserWarning: ... sm_120 is not compatible with the current PyTorch installation` (RTX 5090) | You installed the `cu124` build. Reinstall from the `cu128` index — see §0.1 "sm_120 not compatible" |
| `CUDA error: no kernel image is available for execution on the device` | Same root cause as above — the installed torch build has no kernels for your GPU's compute capability. See §0.1 |
| `.venv/bin/python -m pip ...` → `No module named pip` | `uv venv` doesn't ship pip. Use `uv pip <cmd>` (uninstall/install/list) instead |
| Reinstalling torch with a different `--index-url` doesn't change anything (`Checked 1 package`) | `uv pip install torch` alone won't replace an already-installed `torch`. Run `uv pip uninstall torch torchvision torchaudio` first, or add `--reinstall` |
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

## 9. Verified on the RTX 5090 — measured timings

Pre-flight (`--sample 20000`, `--zeroshot-max 20000`) ran end to end on the 5090 with `cu128` torch. Detected as `NVIDIA GeForce RTX 5090 (31.4 GB VRAM)`, auto batch size **512** for both `base` and `large` model classes, no OOM, no fallback.

**The wall clock was almost entirely model downloads, not compute.** ~3.8 GB of HuggingFace weights at 1.5–3.6 MB/s ≈ 21 min; actual GPU compute across every stage was **under a minute**. Those weights are cached now, so the full run will not pay this again.

| Stage | n (20K sample) | Measured throughput | Compute time | Extrapolated to full corpus |
|---|---|---|---|---|
| `01_preprocess` | 732,194 → 20,000 | — | 38 s (35 s of it the one-time fastText download) | ~2–4 min |
| `02a_vader` (CPU) | 9,409 | 16,226 rev/s | 0.6 s | ~21 s |
| `02b_roberta` en | 8,137 | 8,487 item/s | 1.0 s | ~40 s |
| `02b_roberta` multi | 1,272 | 7,927 item/s | 0.2 s | ~6 s |
| `03_absa` segmentation (CPU) | 9,409 → 17,099 sent | 4,617 rev/s | 2.0 s | ~75 s |
| `03_absa` NLI | 44,595 pairs | 1,612 pair/s | 28 s | **~1.6M pairs ≈ 17 min** |
| `03_absa` aspect sentiment | 6,988 unique | 17,249 item/s | 0.4 s | ~15 s |
| `03b_promise` | 26 descriptions | — | 7 s | 7 s (fixed — independent of `--sample`) |
| `03c_demand` | 9,409 | — | ~1 s | ~40 s |

**Full-corpus GPU stages should total roughly 25 min**, dominated by `03`'s NLI pass. Budget hours only if `04_topic_modeling` is slow — BERTopic's UMAP + HDBSCAN are **CPU-bound and do not use the GPU**, and they scale worse than linearly on ~300K documents. That is the one stage whose full-scale cost is still unmeasured, and it is now the likely bottleneck of the whole pipeline, not the transformers.

Correctness checks from the guide's pre-flight list:

| Check | Result |
|---|---|
| Torch sees `sm_120` and runs real kernels | ✅ after switching to `cu128` (§0.1) |
| `03_absa` lexicon hit rate ≈ 35% | ✅ **35.5%** — 8,312 pairs over 6,070 / 17,099 sentences |
| Zero-shot prefilter working | ✅ 44,595 NLI pairs vs 240,813 unfiltered (**5.4× reduction**) |
| Aspect coverage | ✅ all 27 aspects fire; rare ones present (`qasr_travel` 35, `women_period` 74, `madhab` 29) |
| Request/complaint split | ✅ 757 (7.5%) flagged as requests, excluded from aspect sentiment |
| `03b` mean κ | ✅ **0.257** with `--no-zero-shot` (0.202 with it on — see §5A) |
| `03b` feature_count source bias | ✅ fixed — desc-derived apps mean **8.5** vs csv **9.0** (was 13.0 vs 9.0 with zero-shot on) |
| **Check 1** — RQ2 predictors survive | ✅ `prayer_times_accuracy`, `qibla`, `ui_design` all estimated; no `RQ2 VERDICT UNAVAILABLE`. Only `madhab` (6 cases) and `calc_method` (15) dropped, both below the 25-case floor and expected to clear at full scale |
| **Check 2** — coefficient magnitudes | ✅ max &#124;β&#124; = 1.28 stars (M1 `ads_intrusive`), all well under 2 |
| **Check 2** — no `NaN` p-values | ❌ **M4 / `complaint_complexity_bloat` only** — see §9.2 below |
| **Check 3** — lexicon hit rate | ✅ (above) |
| `04`, `05`, `08` at full scale | ⏳ |

### 9.2 Known degeneracies at 20K — expected to clear at full scale

Two models fail at the sample size, both from sparsity rather than a code defect. **Re-check both after the full run**; if either persists, the fix is a specification change, not a bigger sample.

**M4 / `complaint_complexity_bloat`** — `feature_count` and `ad_supported` return `SE = nan, p = nan`, and `log_installs` returns `SE = 2520.3` with a CI of `[-4939.8, +4939.8]`. That is the separation signature the pre-flight check warns about. Cause: the outcome's prevalence is **0.77% (~72 positive cases)**, and *all three predictors in the formula are app-level constants* fitted alongside a per-app random intercept ([06_models.py:518-522](src/scripts/06_models.py#L518-L522)) — the random intercept absorbs the same between-app variance the fixed effects need, and 26 groups × 72 positives cannot identify both. The identical specification succeeds on `complaint_stability_bugs` (prevalence 2.91%, ~274 positives, `SE = 0.0020`, `p = 0.0056`), which confirms it is prevalence-driven. At full scale `complexity_bloat` should reach ~2,600 positives, comfortably past where the other outcome already works.

> If it still returns `nan` at full scale, drop the random intercept for M4 and use OLS with cluster-robust SEs by app. With every predictor app-level, the random intercept is competing with the fixed effects rather than helping.

**M5 / `mosque_finder`** — `MixedLM failed to converge: Singular matrix`, on 80 mentions across 18 apps. Same story, same remedy.

### 9.0 Verified earlier on the 3050 laptop (correctness only)

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
