# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Phase 1C — Cross-method Sentiment Validation
#
# Compares VADER / RoBERTa / star-proxy against each other and against a
# hand-labelled gold set. See `implementation_plan.md` §4.4.
#
# **This notebook has two modes:**
#
# 1. `--generate` — builds the stratified labelling sheets (500 document-level
#    + 300 aspect-level rows). Run this FIRST, then label them by hand.
# 2. default — reads the filled sheets back and reports κ, precision/recall/F1,
#    and Krippendorff's α.
#
# Open as a notebook with: `jupytext --to notebook 02c_cross_validation.py`

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Locate src/scripts whether this runs as a script or inside a Jupyter kernel.
# `__file__` is defined only in the former — referencing it directly in a
# notebook raises NameError before any cell can run — and the working directory
# differs depending on where Jupyter was started, so try the candidates in turn
# and take the first that actually holds _common.py.
_here = Path(globals()["__file__"]).resolve().parent if "__file__" in globals() else Path.cwd()
for _cand in (_here.parent / "scripts", _here / "src" / "scripts",
              Path.cwd().parent / "scripts", Path.cwd() / "src" / "scripts",
              Path.cwd() / "scripts"):
    if (_cand / "_common.py").is_file():
        sys.path.insert(0, str(_cand))
        break
else:
    raise ImportError(
        "Cannot find src/scripts/_common.py. Start Jupyter from the repo root "
        "or from src/notebooks/, or set the working directory with os.chdir()."
    )

from _common import (  # noqa: E402
    ASPECT_PATH, GOLD_DIR, SENTIMENT_PATH, log, stratified_sample, write_gold_sheet,
)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

DOC_GOLD = GOLD_DIR / "doc_500.csv"
ASPECT_GOLD = GOLD_DIR / "aspect_300.csv"

# %% [markdown]
# ## 1. Load scored corpus

# %%
df = pd.read_parquet(SENTIMENT_PATH)
df = df[df["corpus_text"]].copy()
print(f"C_text: {len(df):,} reviews across {df['app_name'].nunique()} apps")

labelled = df.dropna(subset=["vader_label", "roberta_label", "star_label"])
print(f"Rows with all three labels: {len(labelled):,}")

# %% [markdown]
# ## 2. Pairwise agreement (κ + confusion matrices)
#
# §4.4. Note that VADER is English-tuned, so the headline κ should be reported
# on `C_en` — the all-language number is shown alongside to make the gap visible.

# %%
from sklearn.metrics import cohen_kappa_score, confusion_matrix  # noqa: E402

LABELS = ["Negative", "Neutral", "Positive"]
PAIRS = [("vader_label", "roberta_label"), ("vader_label", "star_label"),
         ("roberta_label", "star_label")]


def agreement_table(data: pd.DataFrame, note: str) -> pd.DataFrame:
    rows = []
    for a, b in PAIRS:
        sub = data.dropna(subset=[a, b])
        if sub.empty:
            continue
        rows.append({
            "pair": f"{a.replace('_label','')} vs {b.replace('_label','')}",
            "n": len(sub),
            "raw_agreement": (sub[a] == sub[b]).mean(),
            "cohens_kappa": cohen_kappa_score(sub[a], sub[b], labels=LABELS),
            "subset": note,
        })
    return pd.DataFrame(rows)


agree = pd.concat([
    agreement_table(labelled, "all languages"),
    agreement_table(labelled[labelled["lang"] == "en"], "C_en only"),
], ignore_index=True)
display(agree.round(3))  # noqa: F821

# %%
for a, b in PAIRS:
    sub = labelled.dropna(subset=[a, b])
    cm = confusion_matrix(sub[a], sub[b], labels=LABELS)
    print(f"\n{a} (rows) × {b} (cols)")
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS))

# %% [markdown]
# ### The RQ2-relevant divergence
#
# §4.3: a 5★ review containing "only complaint is the qibla is off" is
# mislabelled at the aspect level by the star proxy. Quantify that here — it is
# the empirical justification for doing ABSA at all.

# %%
diverge = labelled[
    (labelled["star_label"] == "Positive") & (labelled["roberta_label"] == "Negative")
]
print(f"High-star / negative-text reviews: {len(diverge):,} "
      f"({len(diverge)/len(labelled):.2%} of labelled)")

if ASPECT_PATH.exists():
    asp = pd.read_parquet(ASPECT_PATH)
    neg_aspects = asp[
        (asp["reviewId"].isin(diverge["reviewId"])) & (asp["sentiment_label"] == "Negative")
    ]
    print("\nWhat those reviews complain about, despite the high star:")
    display(neg_aspects["aspect"].value_counts().head(15))  # noqa: F821

# %% [markdown]
# ## 3. Generate the gold-set labelling sheets
#
# **ACTION (§12.2)**: run this once, then fill the sheets by hand.
#
# - `doc_500.csv` — 500 reviews stratified by install tier × star × language,
#   labelled {Positive, Negative, Neutral, Mixed}
# - `aspect_300.csv` — 300 review-sentences labelled for aspect + per-aspect
#   sentiment. §4.4 is explicit that Phase 2 cannot be validated by a
#   document-level gold set alone.
#
# Two annotators on a 100-row overlap gives a reportable Krippendorff's α at
# modest cost; set `N_ANNOTATORS = 2` to emit split files.

# %%
N_ANNOTATORS = 2
OVERLAP = 100


def install_tier(n) -> str:
    if pd.isna(n):
        return "unknown"
    n = float(n)
    if n > 100_000_000:
        return ">100M"
    if n > 10_000_000:
        return "10M-100M"
    if n > 1_000_000:
        return "1M-10M"
    return "<1M"


def star_band(s) -> str:
    if pd.isna(s):
        return "unknown"
    s = float(s)
    return "1-2" if s <= 2 else "3" if s == 3 else "4-5"


def generate_doc_gold(data: pd.DataFrame, n: int = 500, seed: int = 42) -> pd.DataFrame:
    d = data.copy()
    d["install_tier"] = d["realInstalls"].map(install_tier)
    d["star_band"] = d["score"].map(star_band)
    d["lang_band"] = np.where(d["lang"] == "en", "en", "translated")
    d["stratum"] = d["install_tier"] + "|" + d["star_band"] + "|" + d["lang_band"]

    sample = stratified_sample(d, "stratum", n, seed=seed)  # noqa: F405

    sheet = sample[["reviewId", "app_name", "score", "lang", "install_tier",
                    "star_band", "content_clean"]].copy()
    sheet["gold_label"] = ""       # ACTION: Positive / Negative / Neutral / Mixed
    sheet["annotator"] = ""
    sheet["notes"] = ""
    return sheet.reset_index(drop=True)


doc_sheet = generate_doc_gold(df, n=500)
print(f"Document gold set: {len(doc_sheet)} rows, "
      f"{doc_sheet['stratum'].nunique() if 'stratum' in doc_sheet else '-'} strata")
display(doc_sheet.groupby(["install_tier", "star_band"]).size().unstack(fill_value=0))  # noqa: F821

# %%
GOLD_DIR.mkdir(parents=True, exist_ok=True)
write_gold_sheet(doc_sheet, DOC_GOLD, ["gold_label", "annotator", "notes"])
print(f"Wrote {DOC_GOLD}")

if N_ANNOTATORS == 2:
    overlap = doc_sheet.head(OVERLAP)
    rest = doc_sheet.iloc[OVERLAP:]
    half = len(rest) // 2
    for i, part in enumerate([rest.iloc[:half], rest.iloc[half:]], start=1):
        out = pd.concat([overlap.assign(is_overlap=True), part.assign(is_overlap=False)])
        path = write_gold_sheet(
            out, GOLD_DIR / f"doc_500_annotator{i}.csv",
            ["gold_label", "annotator", "notes"],
        )
        print(f"Wrote {path}  ({len(out)} rows, {OVERLAP} shared)")

# %% [markdown]
# ### Aspect-level gold set (300 rows)

# %%
if ASPECT_PATH.exists():
    asp = pd.read_parquet(ASPECT_PATH)
    scored = asp[asp["triggering_sentence"].notna()]
    # Stratify by aspect so rare aspects (women_period, qasr_travel) are covered —
    # a proportional sample would contain almost none of them.
    aspect_sheet = stratified_sample(scored, "aspect", 300, seed=42)
    aspect_sheet = aspect_sheet[[
        "reviewId", "app_name", "aspect", "triggering_sentence",
        "sentiment_label", "match_method", "is_request", "score",
    ]].rename(columns={
        "sentiment_label": "predicted_sentiment", "aspect": "predicted_aspect",
    })
    aspect_sheet["gold_aspect_correct"] = ""    # ACTION: 1 / 0
    aspect_sheet["gold_aspect_true"] = ""       # ACTION: correct aspect if 0
    aspect_sheet["gold_sentiment"] = ""         # ACTION: Positive/Negative/Neutral/Mixed
    aspect_sheet["gold_is_request"] = ""        # ACTION: 1 / 0
    aspect_sheet["annotator"] = ""

    write_gold_sheet(
        aspect_sheet, ASPECT_GOLD,
        ["gold_aspect_correct", "gold_aspect_true", "gold_sentiment",
         "gold_is_request", "annotator"],
    )
    print(f"Wrote {ASPECT_GOLD}  ({len(aspect_sheet)} rows, "
          f"{aspect_sheet['predicted_aspect'].nunique()} aspects)")
else:
    print("aspect_sentiments.parquet not found — run 03_absa.py before generating "
          "the aspect gold set.")

# %% [markdown]
# ## 4. Evaluate against the filled gold sets
#
# Re-run this section after labelling. It is a no-op until `gold_label` is filled.

# %%
from sklearn.metrics import classification_report  # noqa: E402


def evaluate_doc_gold() -> None:
    if not DOC_GOLD.exists():
        print("No gold file yet.")
        return
    gold = pd.read_csv(DOC_GOLD)
    gold = gold[gold["gold_label"].astype(str).str.strip() != ""]
    if gold.empty:
        print("doc_500.csv exists but gold_label is unfilled — nothing to evaluate yet.")
        return

    merged = gold.merge(
        df[["reviewId", "vader_label", "roberta_label", "star_label"]],
        on="reviewId", how="left",
    )
    # 'Mixed' has no counterpart in the three-class predictions; report it
    # separately rather than forcing it into one of the three.
    mixed = merged[merged["gold_label"] == "Mixed"]
    three = merged[merged["gold_label"].isin(LABELS)]
    print(f"Gold rows: {len(merged)}  (three-class {len(three)}, Mixed {len(mixed)})")

    for method in ("vader_label", "roberta_label", "star_label"):
        sub = three.dropna(subset=[method])
        if sub.empty:
            continue
        print(f"\n=== {method} ===")
        print(classification_report(sub["gold_label"], sub[method],
                                    labels=LABELS, zero_division=0))

    if len(mixed):
        print("\nHow each method handles 'Mixed' gold reviews:")
        for method in ("vader_label", "roberta_label", "star_label"):
            print(f"  {method}: {mixed[method].value_counts().to_dict()}")


evaluate_doc_gold()

# %% [markdown]
# ### Krippendorff's α between annotators

# %%
def annotator_alpha() -> None:
    files = sorted(GOLD_DIR.glob("doc_500_annotator*.csv"))
    if len(files) < 2:
        print("Need two filled annotator files for α.")
        return
    frames = []
    for i, f in enumerate(files):
        g = pd.read_csv(f)
        g = g[g["gold_label"].astype(str).str.strip() != ""]
        frames.append(g.set_index("reviewId")["gold_label"].rename(f"a{i}"))
    joined = pd.concat(frames, axis=1, join="inner")
    if joined.empty:
        print("No overlapping labelled rows yet.")
        return
    print(f"Overlapping labelled rows: {len(joined)}")
    print(f"Raw agreement: {(joined.iloc[:, 0] == joined.iloc[:, 1]).mean():.1%}")
    print(f"Cohen's κ: {cohen_kappa_score(joined.iloc[:, 0], joined.iloc[:, 1]):.3f}")
    try:
        import krippendorff
        codes = {lab: i for i, lab in enumerate(sorted(joined.stack().unique()))}
        matrix = joined.replace(codes).T.to_numpy(dtype=float)
        alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement="nominal")
        print(f"Krippendorff's α: {alpha:.3f}")
    except ImportError:
        print("`pip install krippendorff` to report α.")


annotator_alpha()

# %% [markdown]
# ### Zero-shot threshold calibration
#
# §5.3 step 3 requires the zero-shot acceptance threshold be *calibrated* on the
# aspect gold set, not guessed. Run after `aspect_300.csv` is filled, then pass
# the chosen value to `03_absa.py --zeroshot-threshold`.

# %%
def calibrate_zeroshot_threshold() -> None:
    if not ASPECT_GOLD.exists():
        print("No aspect gold file yet.")
        return
    gold = pd.read_csv(ASPECT_GOLD)
    gold = gold[gold["gold_aspect_correct"].astype(str).str.strip() != ""]
    zs = gold[gold["match_method"] == "zero-shot"] if "match_method" in gold else gold
    if zs.empty or "zeroshot_score" not in zs.columns:
        print("No labelled zero-shot rows with scores — precision by method instead:")
        if len(gold) and "match_method" in gold:
            print(gold.groupby("match_method")["gold_aspect_correct"]
                  .astype(float).mean().rename("precision"))
        return
    for t in np.arange(0.5, 0.96, 0.05):
        sel = zs[zs["zeroshot_score"] >= t]
        if len(sel) < 5:
            continue
        prec = sel["gold_aspect_correct"].astype(float).mean()
        print(f"  threshold {t:.2f}: n={len(sel):>4d}  precision={prec:.2%}")


calibrate_zeroshot_threshold()

# %% [markdown]
# ## Next
#
# 1. Fill `data/gold_labels/doc_500.csv` (and the per-annotator splits).
# 2. Fill `data/gold_labels/aspect_300.csv` after `03_absa.py` has run.
# 3. Re-run sections 4 for the numbers that go into the Methods section and Figure 2.
