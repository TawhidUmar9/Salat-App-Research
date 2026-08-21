"""
Phase 2.5a — Promise Extraction
================================
Extract claimed features from Play Store descriptions for all 26 apps.
This covers the 6 unannotated apps and provides a second measurement
for inter-source agreement (Cohen's κ) against the 20 hand-annotated apps.

Input:
    - reviews/*/metadata.json          (26 app descriptions)
    - Salah App Analysis - Sheet1.csv  (20 hand-annotated apps)
    - data/aspect_lexicon.yaml         (shared taxonomy)

Output:
    - data/feature_matrix_combined.parquet
      Schema: app × feature × {csv_annotated, description_claimed, source_agreement}

See implementation_plan.md §6.1 for full specification.

Usage:
    python src/scripts/03b_promise_extraction.py
    python src/scripts/03b_promise_extraction.py --no-zero-shot
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    ASPECT_DESCRIPTIONS, ASPECT_TO_FEATURE, FEATURE_COLUMNS, FEATURE_MATRIX_PATH,
    auto_batch_size, base_parser, compile_aspect_patterns, get_device,
    load_feature_csv, load_lexicon, load_metadata, log, section, set_seed,
)

#: Feature-bearing aspects only — the 20 that map to a CSV column (§5.1).
PROMISE_ASPECTS = list(ASPECT_TO_FEATURE.keys())

#: Split a store description into bullets / sentences.
SEGMENT_RE = re.compile(r"[\n\r]+|(?<=[.!?])\s+|\s*[•·▪◆★✓✔➤➔–—-]\s+")

#: Accept a zero-shot claim only above this entailment probability. Descriptions
#: are marketing copy — a low threshold produces claims for everything.
ZS_THRESHOLD = 0.85


def load_app_descriptions() -> pd.DataFrame:
    """One row per app: identity, description text, and monetization metadata."""
    section("§6.1  Loading app descriptions")

    meta = load_metadata()
    meta["desc_full"] = (meta["summary"].fillna("") + "\n" + meta["description"].fillna("")).str.strip()
    meta["desc_chars"] = meta["desc_full"].str.len()

    log(f"{len(meta)} apps; description length median {meta['desc_chars'].median():.0f} chars "
        f"(range {meta['desc_chars'].min()}–{meta['desc_chars'].max()})")
    empty = meta[meta["desc_chars"] < 200]
    if len(empty):
        log(f"Apps with very short descriptions: {list(empty['app_name'])}", level="WARN")
    return meta


def segment_description(description: str) -> list[str]:
    """Split a description into feature bullets / sentences worth classifying."""
    if not description:
        return []
    segments = [s.strip(" \t·•-–—") for s in SEGMENT_RE.split(description)]
    return [s for s in segments if len(s.split()) >= 3]


def classify_features(
    segments: list[str], patterns: dict, *, scorer=None,
) -> dict[str, tuple[bool, str]]:
    """
    Classify description segments against the 20-feature taxonomy.

    Lexicon matching runs first; the zero-shot model (if supplied) is applied
    only to segments no keyword matched, exactly as in §5.3. Returns
    {aspect: (claimed, evidence_sentence)}.
    """
    result: dict[str, tuple[bool, str]] = {a: (False, "") for a in PROMISE_ASPECTS}
    if not segments:
        return result

    unmatched_idx = set(range(len(segments)))
    for i, seg in enumerate(segments):
        low = seg.lower()
        for aspect in PROMISE_ASPECTS:
            pat = patterns.get(aspect)
            if pat and pat.search(low):
                unmatched_idx.discard(i)
                if not result[aspect][0]:
                    result[aspect] = (True, seg[:300])

    if scorer is not None and unmatched_idx:
        residual = [segments[i] for i in sorted(unmatched_idx)]
        scores = scorer.score(residual)
        names = scorer.label_names
        for row, seg in zip(scores, residual):
            for j, val in enumerate(row):
                aspect = names[j]
                if val >= ZS_THRESHOLD and not result[aspect][0]:
                    result[aspect] = (True, seg[:300])
    return result


def build_description_matrix(meta: pd.DataFrame, patterns: dict, scorer=None) -> pd.DataFrame:
    """Long-format promised[app, feature] with the supporting sentence retained."""
    section("§6.1  Classifying description segments")

    rows = []
    for _, app in meta.iterrows():
        segments = segment_description(app["desc_full"])
        claims = classify_features(segments, patterns, scorer=scorer)
        for aspect, (claimed, evidence) in claims.items():
            rows.append({
                "app_name": app["app_name"],
                "app_key": app["app_key"],
                "package": app["package"],
                "aspect": aspect,
                "feature": ASPECT_TO_FEATURE[aspect],
                "description_claimed": bool(claimed),
                "description_evidence": evidence,
                "n_segments": len(segments),
            })
    out = pd.DataFrame(rows)

    per_app = out.groupby("app_name", observed=True)["description_claimed"].sum().sort_values(ascending=False)
    log("Features claimed per app (description-derived):")
    for app, n in per_app.items():
        log(f"    {str(app)[:44]:44s} {int(n):>2d}/{len(PROMISE_ASPECTS)}")
    return out


def validate_against_csv(description_matrix: pd.DataFrame, csv_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    §6.1 — Inter-source agreement.

    Cohen's κ per feature between the hand annotation and the description
    extraction, computed over the annotated apps only. This is a genuine
    measurement-validity contribution and belongs in the Methods section
    (Figure 3), so it is written out rather than only logged.
    """
    section("§6.1  Validation against hand annotation (Cohen's κ)")

    annotated = csv_matrix[csv_matrix["is_annotated"]]
    log(f"Comparing against {len(annotated)} hand-annotated apps.")

    long_csv = annotated.melt(
        id_vars=["app_key", "app_name_csv"],
        value_vars=[c for c in FEATURE_COLUMNS if c in annotated.columns],
        var_name="feature", value_name="csv_annotated",
    )

    merged = description_matrix.merge(
        long_csv, on=["app_key", "feature"], how="inner", validate="one_to_one",
    )
    if merged.empty:
        log("No apps joined between description matrix and CSV — check app_key normalization.", level="WARN")
        return pd.DataFrame()

    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        log("scikit-learn missing; skipping κ.", level="WARN")
        return pd.DataFrame()

    rows = []
    for feature, grp in merged.groupby("feature", observed=True):
        a = grp["csv_annotated"].astype(float).values
        b = grp["description_claimed"].astype(float).values
        valid = ~np.isnan(a)
        a, b = a[valid], b[valid]
        if len(a) < 2 or (len(set(a)) == 1 and len(set(b)) == 1 and a[0] == b[0]):
            kappa = np.nan     # κ is undefined when both raters are constant
        else:
            kappa = cohen_kappa_score(a.astype(int), b.astype(int))
        rows.append({
            "feature": feature,
            "n_apps": len(a),
            "csv_present": int(a.sum()),
            "description_present": int(b.sum()),
            "agreement": float((a == b).mean()) if len(a) else np.nan,
            "cohens_kappa": kappa,
            "description_misses": int(((a == 1) & (b == 0)).sum()),
            "description_overclaims": int(((a == 0) & (b == 1)).sum()),
        })

    report = pd.DataFrame(rows).sort_values("cohens_kappa", ascending=False)
    log(f"{'feature':34s} {'n':>3s} {'csv':>4s} {'desc':>5s} {'agree':>6s} {'kappa':>6s}")
    for _, r in report.iterrows():
        k = f"{r['cohens_kappa']:.2f}" if pd.notna(r["cohens_kappa"]) else "  n/a"
        log(f"    {r['feature'][:32]:32s} {r['n_apps']:>3.0f} {r['csv_present']:>4.0f} "
            f"{r['description_present']:>5.0f} {r['agreement']:>6.0%} {k:>6s}")

    mean_k = report["cohens_kappa"].mean(skipna=True)
    log("")
    log(f"Mean Cohen's κ across features: {mean_k:.3f}")
    log("Interpretation: <0.20 poor · 0.21–0.40 fair · 0.41–0.60 moderate · "
        "0.61–0.80 substantial · >0.80 near-perfect")

    out_path = FEATURE_MATRIX_PATH.parent / "promise_source_agreement.csv"
    report.to_csv(out_path, index=False)
    log(f"Saved κ report → {out_path}")
    return report


def check_monetization_claims(meta: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    """
    §6.1 — Monetization cross-check.

    'All Features for free' is the one feature the store metadata can
    contradict directly: an app claiming free features while shipping ads and
    IAP is a measurable delivery gap, and feeds RQ6a.
    """
    section("§6.1  Monetization claim check")

    mon = matrix[matrix["aspect"] == "monetization"].merge(
        meta[["app_key", "adSupported", "offersIAP", "free"]], on="app_key", how="left",
    )
    mon["monetized"] = mon["adSupported"] | mon["offersIAP"]
    mon["claims_free_but_monetized"] = mon["description_claimed"] & mon["monetized"]

    flagged = mon[mon["claims_free_but_monetized"]]
    log(f"Apps claiming free features while carrying ads or IAP: {len(flagged)}")
    for _, r in flagged.iterrows():
        log(f"    {str(r['app_name'])[:42]:42s} ads={r['adSupported']} iap={r['offersIAP']}")
    return mon[["app_key", "adSupported", "offersIAP", "monetized", "claims_free_but_monetized"]]


def main() -> None:
    p = base_parser(__doc__ or "Phase 2.5a — promise extraction")
    p.add_argument("--no-zero-shot", action="store_true",
                   help="Lexicon matching only; skip the NLI pass over unmatched segments.")
    p.add_argument("--zeroshot-topk", type=int, default=0,
                   help="Prefilter width. 0 scores all 20 features — descriptions are "
                        "few enough that the full pass is cheap.")
    args = p.parse_args()
    set_seed(args.seed)

    meta = load_app_descriptions()
    lexicon = load_lexicon()
    # variant="promise": store copy is marketing prose, not user complaint, and
    # needs the stricter vocabulary. See compile_aspect_patterns' docstring.
    patterns = compile_aspect_patterns(
        {k: v for k, v in lexicon.items() if k in PROMISE_ASPECTS}, variant="promise",
    )

    scorer = None
    if not args.no_zero_shot:
        from _nlp import ZeroShotScorer
        device, vram = get_device(args.device)
        scorer = ZeroShotScorer(
            labels={a: ASPECT_DESCRIPTIONS[a] for a in PROMISE_ASPECTS},
            device=device,
            batch_size=auto_batch_size(vram, "large", requested=args.batch_size),
            top_k=args.zeroshot_topk or len(PROMISE_ASPECTS),
            fp32=args.fp32,
            # Store copy makes claims; reviews discuss experiences.
            hypothesis_template="This app offers {}.",
        ).load()

    desc_matrix = build_description_matrix(meta, patterns, scorer)
    csv_matrix = load_feature_csv()
    report = validate_against_csv(desc_matrix, csv_matrix)
    mon = check_monetization_claims(meta, desc_matrix)

    section("Combining sources")
    long_csv = csv_matrix.melt(
        id_vars=["app_key", "app_name_csv", "is_annotated"],
        value_vars=[c for c in FEATURE_COLUMNS if c in csv_matrix.columns],
        var_name="feature", value_name="csv_annotated",
    )
    combined = desc_matrix.merge(long_csv, on=["app_key", "feature"], how="left")
    combined["is_annotated"] = combined["is_annotated"].fillna(False)

    # CSV is primary where it exists (hands-on verification); description fills
    # the 6 unannotated apps and is the robustness check elsewhere (§2.2).
    combined["feature_present"] = np.where(
        combined["csv_annotated"].notna(),
        combined["csv_annotated"],
        combined["description_claimed"].astype(float),
    )
    combined["promise_source"] = np.where(
        combined["csv_annotated"].notna(), "csv", "description",
    )
    combined["source_agreement"] = np.where(
        combined["csv_annotated"].isna(), pd.NA,
        combined["csv_annotated"] == combined["description_claimed"].astype(float),
    )
    combined = combined.merge(mon, on="app_key", how="left")

    counts = (
        combined.groupby("app_name", observed=True)["feature_present"]
        .sum().sort_values(ascending=False)
    )
    log("feature_count per app (CSV primary, description fallback):")
    for app, n in counts.items():
        src = combined.loc[combined["app_name"] == app, "promise_source"].iloc[0]
        log(f"    {str(app)[:44]:44s} {n:>4.0f}/20   [{src}]")

    combined.to_parquet(FEATURE_MATRIX_PATH, index=False, compression="zstd")
    log(f"Saved → {FEATURE_MATRIX_PATH}")
    if len(report):
        log(f"Mean κ (CSV vs description): {report['cohens_kappa'].mean(skipna=True):.3f}")
    log("")
    log("Next: python src/scripts/03c_demand_mining.py")


if __name__ == "__main__":
    main()
