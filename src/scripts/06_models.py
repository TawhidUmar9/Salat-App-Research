"""
Phase 5 — Statistical Modelling
=================================
Review-level mixed-effects models (M1–M6) for RQ1–RQ6.
All models use a random intercept per app.

Input:
    - data/master_reviews_with_sentiment.parquet
    - data/aspect_sentiments.parquet
    - data/feature_matrix_combined.parquet
    - data/demand.parquet

Output:
    - data/app_comparison.csv
    - data/gap_matrix.csv (RQ6)
    - data/model_results.csv (all coefficients, with BH-corrected q-values)

See implementation_plan.md §9 for full specification.

Usage:
    python src/scripts/06_models.py
    python src/scripts/06_models.py --models m1,m2
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    ASPECT_PATH, ASPECT_TO_FEATURE, DATA_DIR, DEMAND_PATH, FEATURE_MATRIX_PATH,
    SENTIMENT_PATH,
    base_parser, log, require, section, set_seed,
)

OUTPUT_COMPARISON = DATA_DIR / "app_comparison.csv"
OUTPUT_GAP = DATA_DIR / "gap_matrix.csv"
OUTPUT_RESULTS = DATA_DIR / "model_results.csv"

#: §9.6 — a promised feature counts as a broken promise above this negative
#: rate, provided enough reviews mention it to make the rate meaningful.
BROKEN_PROMISE_NEG_RATE = 0.40
BROKEN_PROMISE_MIN_N = 30

#: Complaint predictors for M1 (§9.1). Split into the accuracy family and the
#: comparison family — RQ2's answer is the contrast between these two blocks.
ACCURACY_ASPECTS = ["prayer_times_accuracy", "qibla", "madhab", "calc_method"]
COMPARISON_ASPECTS = ["ui_design", "stability_bugs", "ads_intrusive", "reminders_adhan"]

_RESULTS: list[dict] = []


def _record(model: str, rq: str, term: str, estimate: float, se: float,
            pvalue: float, n: int, note: str = "", family: str = "primary") -> None:
    """
    Collect one coefficient for the pooled BH correction in §9.7.

    `family` decides whether the coefficient joins the multiple-comparisons
    family. Only "primary" does. Robustness and sensitivity refits describe the
    same hypothesis as their primary model, so counting them again would inflate
    the family and weaken every primary result for no inferential gain — they
    are still written to model_results.csv, just with q_value left NaN.
    """
    _RESULTS.append({
        "model": model, "rq": rq, "term": term, "family": family,
        "estimate": estimate, "std_error": se,
        "ci_low": estimate - 1.96 * se if pd.notna(se) else np.nan,
        "ci_high": estimate + 1.96 * se if pd.notna(se) else np.nan,
        "p_value": pvalue, "n": n, "note": note,
    })


#: A binary predictor with fewer positives than this cannot be estimated
#: reliably; below it you get quasi-separation, absurd coefficients and a
#: singular Hessian whose NaN standard errors contaminate EVERY other
#: coefficient in the same model.
MIN_PREDICTOR_POSITIVES = 25


def _drop_sparse_predictors(
    data: pd.DataFrame, terms: list[str], *, minimum: int = MIN_PREDICTOR_POSITIVES,
) -> list[str]:
    """
    Drop binary predictors with too few positive cases to estimate.

    This is not cosmetic. In testing, `complaint_madhab` with a single positive
    case produced beta = -26.3 on a 1-5 star outcome and made the model
    singular, which turned the p-value of every *other* predictor into NaN —
    a silently wrong result rather than a crash. Dropping the offender is the
    honest move; the dropped term is logged so it can be reported as
    "not estimable at this sample size" rather than quietly omitted.
    """
    keep, dropped = [], []
    for t in terms:
        n = int(data[t].sum()) if t in data.columns else 0
        (keep if n >= minimum else dropped).append((t, n))

    for t, n in dropped:
        log(f"DROPPED '{t}': only {n} positive case(s), below the {minimum} needed "
            f"to estimate. Report as not-estimable, do not read as a null result.",
            level="WARN")
    if dropped:
        log(f"Kept {len(keep)}/{len(terms)} predictors. If this is a --sample run, "
            f"these will very likely be estimable on the full corpus.", level="WARN")
    return [t for t, _ in keep]


def _fit_mixedlm(formula: str, data: pd.DataFrame, group: str = "app_name"):
    """
    Fit a MixedLM, returning None (with a logged reason) rather than raising.

    Convergence warnings are *recorded and surfaced*, not suppressed. statsmodels
    routinely returns a fitted object alongside a ConvergenceWarning ("MLE may be
    on the boundary", "Gradient optimization failed"); silencing those makes an
    unconverged fit indistinguishable from a clean one, and its coefficients
    would go into the paper looking authoritative. A NaN standard error is the
    same story — it means the term is not estimable, which must be reported as
    such rather than read as a null effect (§9.7).
    """
    import statsmodels.formula.api as smf

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = smf.mixedlm(formula, data=data, groups=data[group])
            res = model.fit(method="lbfgs", maxiter=200)
    except Exception as exc:  # noqa: BLE001
        log(f"MixedLM failed to fit: {exc}", level="WARN")
        log(f"  formula: {formula}  (report as NOT ESTIMABLE, not as a null result)",
            level="WARN")
        return None

    flagged = {
        str(w.message).split("\n")[0]
        for w in caught
        if any(k in str(w.message).lower()
               for k in ("converg", "singular", "boundary", "hessian", "gradient"))
    }
    for msg in sorted(flagged):
        log(f"CONVERGENCE WARNING — {msg}", level="WARN")

    n_nan_se = int(res.bse.isna().sum()) if hasattr(res, "bse") else 0
    if n_nan_se:
        log(f"NOT ESTIMABLE: {n_nan_se} coefficient(s) have NaN standard errors "
            f"(separation / collinearity). Report as not-estimable, never as null.",
            level="WARN")
        log(f"  formula: {formula}", level="WARN")

    return res


#: RQ5's reframed hypothesis, fixed BEFORE the full-corpus run (§6, "RQ5 reframe").
#: If bloat is about how features are surfaced rather than how many exist, then
#: bloat complaints should travel with interface/navigation complaints even while
#: being unrelated to feature_count. `ui_design` is the confirmatory pair; the
#: rest of the lift table is exploratory context, deliberately untested.
BLOAT_SURFACING_PAIR = ("complaint_complexity_bloat", "complaint_ui_design")


def _bloat_surfacing(design: pd.DataFrame) -> None:
    """
    RQ5 reframed — is bloat about surfacing rather than count?

    Two things, kept firmly apart:

    1. CONFIRMATORY. A single pre-specified 2x2 test of whether bloat and
       ui_design complaints co-occur above chance. One test, entered into the
       BH-FDR family like any other primary coefficient.
    2. EXPLORATORY. Lift of every other aspect within bloat reviews, logged for
       interpretation and explicitly NOT tested — 27 aspects fished for
       significance is exactly the practice BH is meant to discipline.
    """
    section("§9.4  RQ5 reframed — is bloat about surfacing, not count?")

    bloat_col, ui_col = BLOAT_SURFACING_PAIR
    missing = [c for c in BLOAT_SURFACING_PAIR if c not in design.columns]
    if missing:
        log(f"{missing} absent from the design matrix — skipping.", level="WARN")
        return

    b = design[bloat_col].astype(bool)
    u = design[ui_col].astype(bool)
    n = len(design)
    n_bloat = int(b.sum())
    log(f"Reviews mentioning bloat: {n_bloat:,} / {n:,} ({n_bloat / n:.2%})")
    if n_bloat < MIN_PREDICTOR_POSITIVES:
        log(f"Only {n_bloat} bloat reviews — NOT ESTIMABLE, report count only.",
            level="WARN")
        return

    table = np.array([[int((b & u).sum()), int((b & ~u).sum())],
                      [int((~b & u).sum()), int((~b & ~u).sum())]])
    p_ui_given_bloat = table[0, 0] / max(1, table[0].sum())
    p_ui_base = int(u.sum()) / n
    lift = p_ui_given_bloat / p_ui_base if p_ui_base else np.nan

    from scipy.stats import fisher_exact
    odds, p = fisher_exact(table)
    log(f"CONFIRMATORY  bloat x ui_design:")
    log(f"    P(ui | bloat) = {p_ui_given_bloat:.2%}   base P(ui) = {p_ui_base:.2%}"
        f"   lift = {lift:.2f}x")
    log(f"    Fisher exact OR = {odds:.2f}, p = {p:.2g}   (n={n:,})")
    log(f"    {'SUPPORTS' if (odds > 1 and p < 0.05) else 'DOES NOT SUPPORT'} "
        f"the surfacing account.")
    _record("M4-reframe", "RQ5", "bloat_x_ui_design_cooccurrence",
            float(np.log(odds)) if odds > 0 and np.isfinite(odds) else np.nan,
            np.nan, float(p), n,
            note=f"Fisher exact log-OR; P(ui|bloat)={p_ui_given_bloat:.3f} "
                 f"base={p_ui_base:.3f} lift={lift:.2f}")

    others = [c for c in design.columns
              if c.startswith("complaint_") and c not in BLOAT_SURFACING_PAIR]
    rows = []
    for c in others:
        v = design[c].astype(bool)
        base = v.mean()
        if base <= 0:
            continue
        rows.append((c.replace("complaint_", ""), int((b & v).sum()),
                     (b & v).sum() / n_bloat, base, (b & v).sum() / n_bloat / base))
    rows.sort(key=lambda r: -r[4])
    log("")
    log("EXPLORATORY — aspect lift within bloat reviews (NOT tested, context only):")
    log(f"    {'aspect':<26}{'n':>7}{'P|bloat':>10}{'base':>9}{'lift':>8}")
    for name, cnt, p_in, base, lf in rows[:10]:
        log(f"    {name:<26}{cnt:>7}{p_in:>9.2%}{base:>9.2%}{lf:>7.2f}x")


def _cluster_robust_ols(formula: str, data: pd.DataFrame, outcome: str,
                        terms: list[str], group: str = "app_name") -> None:
    """
    OLS with cluster-robust (CR1) standard errors, clustered on `group`.

    Companion to the MixedLM for models whose predictors are all group-level
    constants. The point estimates match OLS; only the SEs change, and they stay
    finite where the mixed model's can go NaN. Logged as a sensitivity analysis
    so the paper can report both without either being a post-hoc rescue.
    """
    import statsmodels.formula.api as smf

    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            res = smf.ols(formula, data=data).fit(
                cov_type="cluster", cov_kwds={"groups": data[group]},
            )
    except Exception as exc:  # noqa: BLE001
        log(f"  cluster-robust OLS failed: {exc}", level="WARN")
        return

    n_clusters = data[group].nunique()
    log(f"  sensitivity — OLS, SEs clustered on {group} ({n_clusters} clusters):")
    if n_clusters < 30:
        log(f"    NOTE: {n_clusters} clusters is few for cluster-robust inference; "
            f"SEs are anti-conservative. Report as sensitivity, not as the headline.",
            level="WARN")
    for t in terms:
        if t not in res.params.index:
            continue
        b, se, p = res.params[t], res.bse[t], res.pvalues[t]
        log(f"    {t[:30]:30s} β={b:>+9.4f}  SE={se:.4f}  "
            f"[{b - 1.96 * se:+.4f},{b + 1.96 * se:+.4f}]  p={p:.4f}")
        _record("M4-sens", "RQ5", f"{outcome}:{t}", b, se, p, len(data),
                note=f"OLS cluster-robust on {group} ({n_clusters} clusters)",
                family="sensitivity")


def _log_coefs(res, model: str, rq: str, n: int, terms: list[str] | None = None) -> None:
    if res is None:
        # Leave a visible trace: a model that never fitted must not simply be
        # absent from the log and from model_results.csv, or a reader cannot
        # tell "not estimable" from "never attempted".
        log(f"{model} / {rq}: NO RESULT — model did not fit (n={n:,}). "
            f"Report as not-estimable.", level="WARN")
        return
    params, bse, pvals = res.params, res.bse, res.pvalues
    show = terms or [t for t in params.index if t not in ("Intercept", "Group Var")]
    log(f"{'term':32s} {'β':>9s} {'SE':>8s} {'95% CI':>18s} {'p':>8s}")
    for t in show:
        if t not in params.index:
            continue
        b, se, p = params[t], bse.get(t, np.nan), pvals.get(t, np.nan)
        ci = f"[{b - 1.96 * se:+.3f},{b + 1.96 * se:+.3f}]" if pd.notna(se) else "        n/a"
        log(f"    {t[:30]:30s} {b:>+9.4f} {se:>8.4f} {ci:>18s} {p:>8.4f}")
        _record(model, rq, t, b, se, p, n)


def build_review_design(df: pd.DataFrame, aspect_df: pd.DataFrame) -> pd.DataFrame:
    """
    Review-level design matrix: one row per review, one binary column per
    aspect complaint.

    A "complaint" is a negative-sentiment, non-request mention of that aspect.
    Requests are excluded deliberately (§5.3 step 5): "please add a qibla" is
    demand, not dissatisfaction with an existing qibla.
    """
    section("Building the review-level design matrix")

    complaints = aspect_df[
        (aspect_df["sentiment_label"] == "Negative") & (~aspect_df["is_request"].fillna(False))
    ]
    wide = (
        complaints.assign(v=1)
        .pivot_table(index="reviewId", columns="aspect", values="v",
                     aggfunc="max", fill_value=0)
    )
    wide.columns = [f"complaint_{c}" for c in wide.columns]

    design = df.merge(wide, left_on="reviewId", right_index=True, how="left")
    complaint_cols = [c for c in design.columns if c.startswith("complaint_")]
    design[complaint_cols] = design[complaint_cols].fillna(0).astype("int8")

    mentions = aspect_df[~aspect_df["is_request"].fillna(False)].assign(v=1).pivot_table(
        index="reviewId", columns="aspect", values="v", aggfunc="max", fill_value=0
    )
    mentions.columns = [f"mentions_{c}" for c in mentions.columns]
    design = design.merge(mentions, left_on="reviewId", right_index=True, how="left")
    mention_cols = [c for c in design.columns if c.startswith("mentions_")]
    design[mention_cols] = design[mention_cols].fillna(0).astype("int8")

    design["log_words"] = np.log1p(design["n_words"])
    design["review_year_c"] = design["year"].astype(float) - design["year"].astype(float).mean()
    design["sent_num"] = design["roberta_label"].map(
        {"Negative": -1.0, "Neutral": 0.0, "Positive": 1.0}
    )

    log(f"Design matrix: {len(design):,} reviews × {len(complaint_cols)} complaint columns")
    log("Complaint prevalence (top 10):")
    for c, v in design[complaint_cols].mean().sort_values(ascending=False).head(10).items():
        log(f"    {c:34s} {v:.2%}")
    return design


# ─── M1 — RQ2 ───────────────────────────────────────────────────────────────────

def model_m1_rq2_competing_predictors(design: pd.DataFrame):
    """
    §9.1 — M1: competing predictors of dissatisfaction.

    The paper's RQ2 answer is the contrast between the accuracy-family
    coefficients and complaint_ui_design: which class of complaint costs more
    stars, within the same app, controlling for review length and era.
    """
    section("§9.1  M1 — RQ2: accuracy vs. UI complaints on star rating")

    terms = [f"complaint_{a}" for a in ACCURACY_ASPECTS + COMPARISON_ASPECTS]
    terms = [t for t in terms if t in design.columns]
    if not terms:
        log("No complaint columns available — skipping M1.", level="WARN")
        return None

    d = design.dropna(subset=["score", "log_words", "review_year_c"]).copy()
    d["score"] = d["score"].astype(float)

    terms = _drop_sparse_predictors(d, terms)
    if not terms:
        log("Every complaint predictor was too sparse to fit — skipping M1.", level="WARN")
        return None

    formula = f"score ~ {' + '.join(terms)} + log_words + review_year_c"
    log(f"Formula: {formula} + (1 | app_name)")
    log(f"n = {len(d):,} reviews across {d['app_name'].nunique()} apps")

    res = _fit_mixedlm(formula, d)
    if res is None:
        return None
    _log_coefs(res, "M1", "RQ2", len(d), terms)

    acc = [res.params[t] for t in terms if t.replace("complaint_", "") in ACCURACY_ASPECTS]
    ui = res.params.get("complaint_ui_design", np.nan)
    log("")
    if acc:
        log(f"Accuracy-family mean β: {np.mean(acc):+.4f}   (stars lost per complaint)")
    else:
        log("Accuracy-family: no predictor survived the sparsity filter.", level="WARN")
    if pd.notna(ui):
        log(f"UI complaint β:         {ui:+.4f}")
    else:
        log("UI complaint β:         NOT ESTIMABLE (dropped as too sparse)", level="WARN")

    if pd.notna(ui) and len(acc):
        verdict = "accuracy complaints cost MORE" if np.mean(acc) < ui else "UI complaints cost MORE"
        log(f"→ RQ2: {verdict} stars.")
    else:
        # Refusing to state a verdict is the point: RQ2 *is* this comparison,
        # and half of it is missing.
        log("→ RQ2 VERDICT UNAVAILABLE: the accuracy-vs-UI contrast needs both sides. "
            "Re-run on the full corpus.", level="WARN")
    log("Robustness: refit as an ordinal model — star is ordinal and ceiling-heavy.")

    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        sub = d.sample(min(50_000, len(d)), random_state=0)
        om = OrderedModel(sub["score"].astype(int), sub[terms + ["log_words"]], distr="logit").fit(
            method="bfgs", disp=False
        )
        log("Ordinal robustness check (subsample, no random effect):")
        for t in terms:
            log(f"    {t:34s} β={om.params[t]:+.4f}  p={om.pvalues[t]:.4f}")
            _record("M1-ordinal", "RQ2", t, om.params[t], om.bse[t], om.pvalues[t], len(sub),
                    "ordinal robustness check, no random effect")
    except Exception as exc:  # noqa: BLE001
        log(f"Ordinal model skipped: {exc}", level="WARN")
    return res


# ─── M2 — RQ1 ───────────────────────────────────────────────────────────────────

def model_m2_rq1_gamification_valence(design: pd.DataFrame, aspect_df: pd.DataFrame,
                                      feature_df: pd.DataFrame):
    """
    §9.2 — M2: gamification valence.

    The love/hate tension is the RATIO of guilt-coded to motivation-coded
    affect, not the mean sentiment: a mean can hide a bimodal split, which is
    precisely the phenomenon RQ1 posits. Hartigan's dip test is reported
    alongside the distribution shape.
    """
    section("§9.2  M2 — RQ1: gamification valence")

    tracker_aspects = ["prayer_tracker", "tracker_score", "goal_system"]
    tracker_ids = set(aspect_df.loc[aspect_df["aspect"].isin(tracker_aspects), "reviewId"])
    d = design[design["reviewId"].isin(tracker_ids)].copy()
    log(f"Tracker-aspect reviews: {len(d):,} across {d['app_name'].nunique()} apps")
    if len(d) < 100:
        log("Too few tracker reviews to model.", level="WARN")
        return None

    # Tracker tier from the feature matrix: none / tracker only / tracker + score.
    tier = {}
    for app, grp in feature_df.groupby("app_name", observed=True):
        has = dict(zip(grp["aspect"], grp["feature_present"].fillna(0)))
        if has.get("tracker_score", 0) >= 1:
            tier[app] = "tracker_plus_score"
        elif has.get("prayer_tracker", 0) >= 1:
            tier[app] = "tracker_only"
        else:
            tier[app] = "none"
    d["tracker_tier"] = d["app_name"].map(tier).fillna("none")
    log("Tracker tier distribution:")
    for t, n in d["tracker_tier"].value_counts().items():
        log(f"    {t:20s} {n:>7,} reviews")

    if d["tracker_tier"].nunique() > 1:
        sub = d.dropna(subset=["sent_num"])
        res = _fit_mixedlm("sent_num ~ C(tracker_tier)", sub)
        _log_coefs(res, "M2", "RQ1", len(d))

        # Same structural problem as M4, and for the same reason: tracker_tier is
        # a property of the APP (derived from its feature annotation), so it is
        # constant within every random-intercept group. With 26 apps and a tier
        # distribution as skewed as this one the mixed model separates outright —
        # coefficients in the tens with standard errors in the millions. OLS with
        # SEs clustered on app makes no random-effect assumption and still
        # respects the nesting, so RQ1 has an estimate either way.
        _cluster_robust_ols("sent_num ~ C(tracker_tier)", sub, "sent_num",
                            [c for c in ("C(tracker_tier)[T.tracker_only]",
                                         "C(tracker_tier)[T.tracker_plus_score]")])
    else:
        log("Only one tracker tier present — skipping the tier model.", level="WARN")
        res = None

    # Affect split within tracker reviews vs. the corpus baseline.
    affect = aspect_df[aspect_df["aspect"] == "spiritual_affect"]
    guilt_re = r"\b(guilt|guilty|anxious|anxiety|pressure|ashamed|shame|stress|depress|fail)"
    motiv_re = r"\b(motivat|encourag|discipline|consisten|barakah|closer to allah|helpful|improve)"

    tracker_affect = affect[affect["reviewId"].isin(tracker_ids)]
    text = tracker_affect["triggering_sentence"].fillna("").str.lower()
    guilt = text.str.contains(guilt_re, regex=True).sum()
    motiv = text.str.contains(motiv_re, regex=True).sum()

    base_text = affect["triggering_sentence"].fillna("").str.lower()
    base_guilt = base_text.str.contains(guilt_re, regex=True).sum()
    base_motiv = base_text.str.contains(motiv_re, regex=True).sum()

    log("")
    log("Affect split (spiritual_affect sentences):")
    log(f"    within tracker reviews: guilt={guilt:,}  motivation={motiv:,}  "
        f"ratio={guilt / max(1, motiv):.2f}")
    log(f"    corpus baseline:        guilt={base_guilt:,}  motivation={base_motiv:,}  "
        f"ratio={base_guilt / max(1, base_motiv):.2f}")

    if guilt + motiv > 0 and base_guilt + base_motiv > 0:
        from scipy.stats import fisher_exact
        table = [[guilt, motiv], [base_guilt - guilt, base_motiv - motiv]]
        try:
            odds, p = fisher_exact(table)
            log(f"    Fisher exact: OR={odds:.2f}, p={p:.4g} "
                f"(guilt over-representation in tracker reviews)")
            _record("M2-affect", "RQ1", "guilt_vs_motivation_OR", odds, np.nan, p,
                    guilt + motiv, "Fisher exact vs corpus baseline")
        except Exception as exc:  # noqa: BLE001
            log(f"Fisher test skipped: {exc}", level="WARN")

    # Bimodality: is the tracker sentiment distribution split rather than centred?
    vals = d["sent_num"].dropna().to_numpy()
    log("")
    log(f"Tracker sentiment distribution: mean={vals.mean():+.3f}, "
        f"sd={vals.std():.3f}, %neg={np.mean(vals < 0):.1%}, %pos={np.mean(vals > 0):.1%}")
    try:
        from diptest import diptest
        stat, p = diptest(vals)
        log(f"Hartigan's dip test: D={stat:.4f}, p={p:.4g} "
            f"({'bimodal' if p < 0.05 else 'not significantly bimodal'})")
        _record("M2-dip", "RQ1", "hartigan_dip", stat, np.nan, p, len(vals))
    except ImportError:
        # Fall back to a bimodality coefficient; diptest is an optional dep.
        from scipy.stats import kurtosis, skew
        n = len(vals)
        bc = (skew(vals) ** 2 + 1) / (kurtosis(vals) + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
        log(f"diptest not installed. Bimodality coefficient={bc:.3f} "
            f"(>0.555 suggests bimodality). `pip install diptest` for Hartigan's test.")

    export = d.nsmallest(50, "sent_num")[
        ["reviewId", "app_name", "score", "at", "content_clean"]
    ]
    path = DATA_DIR / "quotes" / "rq1_tracker_negative_50.csv"
    export.to_csv(path, index=False)
    log(f"Wrote 50 tracker-negative reviews for qualitative coding → {path}")
    return res


# ─── M3 — RQ4 ───────────────────────────────────────────────────────────────────

def model_m3_rq4_inclusion(design: pd.DataFrame, aspect_df: pd.DataFrame,
                           feature_df: pd.DataFrame, demand_df: pd.DataFrame):
    """
    §9.3 — M3: bio-spiritual inclusion, with honest power.

    The primary analysis is volume/sentiment/demand, which is well powered.
    The 3-vs-17 between-app comparison is reported as descriptive only — N=3
    precludes causal inference and the caption must say so.
    """
    section("§9.3  M3 — RQ4: inclusion (FemTech)")

    women = aspect_df[aspect_df["aspect"] == "women_period"]
    log(f"women_period mentions: {len(women):,} across "
        f"{women['reviewId'].nunique():,} reviews and {women['app_name'].nunique()} apps")

    if len(women) == 0:
        log("No women_period mentions found. That scarcity is itself the finding (§9.3).",
            level="WARN")
    else:
        dist = women["sentiment_label"].value_counts(normalize=True, dropna=True)
        log("Sentiment on women_period mentions:")
        for lab in ("Positive", "Neutral", "Negative"):
            log(f"    {lab:9s} {dist.get(lab, 0.0):.1%}")
        log("Per-app volume:")
        for app, n in women["app_name"].value_counts().head(15).items():
            log(f"    {str(app)[:40]:40s} {n:>5,}")

    # Demand for menstrual handling among apps lacking the feature.
    if len(demand_df):
        wd = demand_df[demand_df["aspect"] == "women_period"]
        log("")
        log(f"Demand signals for women_period: {len(wd):,} across {wd['app_name'].nunique()} apps")
        for kind, n in wd["request_type"].value_counts().items():
            log(f"    {kind:12s} {n:>5,}")

    # Secondary, underpowered: apps with vs. without the feature.
    has_feature = feature_df[
        (feature_df["aspect"] == "women_period") & (feature_df["feature_present"] >= 1)
    ]["app_name"].unique()
    log("")
    log(f"Apps with Women tracking (N={len(has_feature)}): {list(has_feature)}")
    log("UNDERPOWERED — descriptive only. N=3 precludes causal inference (§9.3).")
    log("Confound to state: Islamic Habit tracker is among these and carries the "
        "lowest CSV rating in the corpus (3.8).")

    d = design.copy()
    d["has_women_feature"] = d["app_name"].isin(has_feature)
    comp = d.groupby("has_women_feature", observed=True).agg(
        n_reviews=("reviewId", "size"),
        n_apps=("app_name", "nunique"),
        mean_score=("score", "mean"),
        mean_sentiment=("sent_num", "mean"),
    )
    log(comp.to_string())

    boot = _bootstrap_diff(
        d.loc[d["has_women_feature"], "score"].dropna().astype(float).to_numpy(),
        d.loc[~d["has_women_feature"], "score"].dropna().astype(float).to_numpy(),
    )
    if boot:
        log(f"Bootstrap mean-score difference: {boot[0]:+.3f} "
            f"[{boot[1]:+.3f}, {boot[2]:+.3f}] (95% CI, app-level clustering NOT modelled)")
        _record("M3", "RQ4", "women_feature_score_diff", boot[0], np.nan, np.nan,
                len(d), "descriptive, N=3 treated apps, underpowered")

    _loyalty_proxies(design, demand_df)
    return comp


def _bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000):
    """Percentile bootstrap CI for a difference in means."""
    if len(a) < 5 or len(b) < 5:
        return None
    rng = np.random.default_rng(0)
    diffs = [
        rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
        for _ in range(n_boot)
    ]
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _loyalty_proxies(design: pd.DataFrame, demand_df: pd.DataFrame) -> pd.DataFrame:
    """
    §9.3 — Loyalty proxies.

    Play data has no retention metric. These are PROXIES and every table and
    figure using them must be labelled as such.
    """
    section("§9.3  Loyalty proxies (label as proxies in the paper)")

    rows = []
    for app, grp in design.groupby("app_name", observed=True):
        tenure = churn = 0
        if len(demand_df):
            sub = demand_df[demand_df["app_name"] == app]
            tenure = int((sub["request_type"] == "tenure").sum())
            churn = int((sub["request_type"] == "churn").sum())

        g = grp.dropna(subset=["at"]).sort_values("at")
        slope = np.nan
        if len(g) > 100:
            x = (g["at"] - g["at"].min()).dt.days.to_numpy(dtype=float)
            y = g["score"].astype(float).to_numpy()
            ok = ~np.isnan(y)
            if ok.sum() > 50 and np.ptp(x[ok]) > 0:
                slope = float(np.polyfit(x[ok], y[ok], 1)[0] * 365)  # stars/year

        rows.append({
            "app_name": app,
            "n_reviews": len(grp),
            "tenure_mentions": tenure,
            "tenure_rate": tenure / len(grp),
            "churn_mentions": churn,
            "churn_rate": churn / len(grp),
            "mean_thumbs_up": grp["thumbsUpCount"].mean(),
            "rating_slope_per_year": slope,
        })

    out = pd.DataFrame(rows).sort_values("churn_rate", ascending=False)
    log(f"{'app':38s} {'n':>7s} {'tenure%':>8s} {'churn%':>7s} {'slope/yr':>9s}")
    for _, r in out.iterrows():
        s = f"{r['rating_slope_per_year']:+.3f}" if pd.notna(r["rating_slope_per_year"]) else "    n/a"
        log(f"    {str(r['app_name'])[:36]:36s} {r['n_reviews']:>7,} "
            f"{r['tenure_rate']:>8.2%} {r['churn_rate']:>7.2%} {s:>9s}")
    return out


# ─── M4 — RQ5 ───────────────────────────────────────────────────────────────────

def model_m4_rq5_bloat_hardware(design: pd.DataFrame, feature_df: pd.DataFrame,
                                demand_df: pd.DataFrame):
    """
    §9.4 — M4: feature bloat vs. companion hardware.

    Confound named explicitly: feature count correlates with install base and
    company size. log(realInstalls) is a required control and residual
    confounding is acknowledged rather than modelled away.
    """
    section("§9.4  M4 — RQ5: feature bloat")

    counts = (
        feature_df.groupby("app_name", observed=True)["feature_present"]
        .sum().rename("feature_count").reset_index()
    )
    log(f"feature_count computed for {len(counts)} apps "
        f"(range {counts['feature_count'].min():.0f}–{counts['feature_count'].max():.0f})")

    d = design.merge(counts, on="app_name", how="left")
    d["log_installs"] = np.log1p(d["realInstalls"].astype(float))
    d["ad_supported"] = d["adSupported"].astype(float)

    for outcome in ("complaint_complexity_bloat", "complaint_stability_bugs"):
        if outcome not in d.columns:
            log(f"{outcome} absent from the design matrix — skipping.", level="WARN")
            continue
        sub = d.dropna(subset=["feature_count", "log_installs", "ad_supported"]).copy()
        log("")
        log(f"Outcome: {outcome}  (prevalence {sub[outcome].mean():.2%}, n={len(sub):,})")
        formula = f"{outcome} ~ feature_count + log_installs + ad_supported"
        # Linear probability model with a random intercept: a logistic MixedLM
        # over ~300K rows does not converge reliably in statsmodels, and the LPM
        # coefficient is directly interpretable as a percentage-point change.
        res = _fit_mixedlm(formula, sub)
        _log_coefs(res, "M4", "RQ5", len(sub), ["feature_count", "log_installs", "ad_supported"])

        # Pre-registered sensitivity analysis, always run — not a fallback.  # noqa: E501
        #
        # Every predictor here is an app-level constant, so the per-app random
        # intercept competes with them for the same between-app variance. With
        # 26 groups and a rare outcome that is close to unidentified, and the
        # MixedLM can return NaN standard errors (see _fit_mixedlm). OLS with
        # SEs clustered on app is the standard remedy: it makes no random-effect
        # assumption and still respects the fact that reviews are nested in apps.
        # Reporting both is what lets RQ5 survive review whichever way the
        # mixed model behaves.
        _cluster_robust_ols(formula, sub, outcome,
                            ["feature_count", "log_installs", "ad_supported"])

    _bloat_surfacing(design)

    # App-level exploratory correlation.
    app_level = (
        design.groupby("app_name", observed=True)
        .agg(mean_sentiment=("sent_num", "mean"), mean_score=("score", "mean"),
             n=("reviewId", "size"))
        .reset_index()
        .merge(counts, on="app_name", how="left")
        .dropna(subset=["feature_count"])
    )
    if len(app_level) >= 5:
        from scipy.stats import spearmanr
        log("")
        log(f"App-level Spearman (n={len(app_level)} apps) — EXPLORATORY ONLY:")
        for col in ("mean_sentiment", "mean_score"):
            rho, p = spearmanr(app_level["feature_count"], app_level[col])
            log(f"    feature_count vs {col:16s} ρ={rho:+.3f}  p={p:.4f}")
            _record("M4-app", "RQ5", f"spearman_{col}", rho, np.nan, p, len(app_level),
                    "app-level, exploratory, n<30")

    _hardware_case_study(design, demand_df)
    return app_level


def _hardware_case_study(design: pd.DataFrame, demand_df: pd.DataFrame) -> None:
    """§9.4 — iQIBLA Life as a named case study, not an anonymous treatment cell."""
    section("§9.4  Companion hardware case study (iQIBLA Life)")

    mask = design["app_name"].astype(str).str.contains("iQIBLA", case=False, na=False)
    hw = design[mask]
    log(f"iQIBLA Life reviews in the corpus: {len(hw):,}")
    if len(hw):
        log(f"    mean star {hw['score'].astype(float).mean():.2f}, "
            f"mean sentiment {hw['sent_num'].mean():+.3f}")
        if "complaint_companion_hardware" in hw.columns:
            log(f"    hardware complaints: {int(hw['complaint_companion_hardware'].sum()):,}")

    if len(demand_df):
        hw_apps = set(design.loc[mask, "app_name"])
        hw_demand = demand_df[demand_df["aspect"] == "companion_hardware"]
        elsewhere = hw_demand[~hw_demand["app_name"].isin(hw_apps)]
        log(f"Wearable/watch demand in the apps WITHOUT companion hardware: "
            f"{len(elsewhere):,} signals across {elsewhere['app_name'].nunique()} apps "
            f"({elsewhere['reviewId'].nunique():,} distinct reviews)")
        log("    → demand evidence for RQ5's hardware half, which N=1 cannot test directly.")


# ─── M5 — RQ3 ───────────────────────────────────────────────────────────────────

def model_m5_rq3_competitive_edge(design: pd.DataFrame, aspect_df: pd.DataFrame,
                                  demand_df: pd.DataFrame, feature_df: pd.DataFrame):
    """
    §9.5 — M5: competitive edge.

    Mosque Finder (8/20) is testable. Auto Qasr (1/20) is not: the *problem*
    is measurable ecosystem-wide even though the *solution* exists in one app,
    so it is reported as demand evidence.
    """
    section("§9.5  M5 — RQ3: competitive edge")

    # Preference citations per app — the operationalization of "edge".
    if len(demand_df):
        pref = demand_df[demand_df["request_type"] == "preference"]
        per_app = pref.groupby("app_name", observed=True).size().rename("preference_citations")
        totals = design.groupby("app_name", observed=True).size().rename("n_reviews")
        rate = pd.concat([per_app, totals], axis=1).fillna(0)
        rate["citation_rate"] = rate["preference_citations"] / rate["n_reviews"].replace(0, np.nan)
        log("Preference-citation rate per app:")
        for app, r in rate.sort_values("citation_rate", ascending=False).head(12).iterrows():
            log(f"    {str(app)[:38]:38s} {r['preference_citations']:>5.0f} / "
                f"{r['n_reviews']:>7,.0f} = {r['citation_rate']:.3%}")

    # Mosque finder: testable between-app comparison.
    has_mf = feature_df[
        (feature_df["aspect"] == "mosque_finder") & (feature_df["feature_present"] >= 1)
    ]["app_name"].unique()
    log("")
    log(f"Apps with Mosque Finder (N={len(has_mf)}) — testable.")

    mf = aspect_df[aspect_df["aspect"] == "mosque_finder"]
    if len(mf):
        d = design[design["reviewId"].isin(set(mf["reviewId"]))].copy()
        d["has_mosque_finder"] = d["app_name"].isin(has_mf)
        log(f"mosque_finder mentions: {len(mf):,} across {mf['app_name'].nunique()} apps")
        if d["has_mosque_finder"].nunique() > 1:
            res = _fit_mixedlm("sent_num ~ has_mosque_finder", d.dropna(subset=["sent_num"]))
            _log_coefs(res, "M5", "RQ3", len(d))

    # Auto Qasr: demand evidence only.
    log("")
    log("Auto Qasr (1/20 apps) — NOT testable between apps; reporting demand instead:")
    has_qasr = feature_df[
        (feature_df["aspect"] == "qasr_travel") & (feature_df["feature_present"] >= 1)
    ]["app_name"].unique()
    log(f"    apps with the feature: {list(has_qasr)}")

    if len(demand_df):
        qd = demand_df[demand_df["aspect"] == "qasr_travel"]
        without = qd[~qd["app_name"].isin(has_qasr)]
        log(f"    qasr/travel demand signals in apps WITHOUT it: {len(without):,} "
            f"across {without['app_name'].nunique()} apps, "
            f"{without['reviewId'].nunique():,} distinct reviews")
        _record("M5-demand", "RQ3", "qasr_demand_without_feature", len(without), np.nan,
                np.nan, len(demand_df), "demand evidence; between-app test not possible")

    qasr_complaints = aspect_df[
        (aspect_df["aspect"] == "qasr_travel") & (aspect_df["sentiment_label"] == "Negative")
    ]
    log(f"    negative qasr/travel mentions ecosystem-wide: {len(qasr_complaints):,}")
    log("    → The PROBLEM is measurable ecosystem-wide even though the SOLUTION "
        "exists in one app (§9.5).")


# ─── M6 — RQ6 ───────────────────────────────────────────────────────────────────

def model_m6_rq6_gap_matrix(feature_df: pd.DataFrame, aspect_df: pd.DataFrame,
                            demand_df: pd.DataFrame) -> pd.DataFrame:
    """
    §9.6 — M6: the gap matrix. The synthesis RQ.

    6a delivery gap: feature claimed → do users complain about it?
    6b coverage gap: feature absent  → do users request it?

    The ranked unmet-need list this produces is the bridge to the paper's
    design-implications section.
    """
    section("§9.6  M6 — RQ6: delivery gap (6a) and coverage gap (6b)")

    mentions = aspect_df[~aspect_df["is_request"].fillna(False)]
    per = (
        mentions.groupby(["app_name", "aspect"], observed=True)
        .agg(
            n_mentions=("reviewId", "nunique"),
            n_negative=("sentiment_label", lambda s: int((s == "Negative").sum())),
        )
        .reset_index()
    )
    per["neg_rate"] = per["n_negative"] / per["n_mentions"]

    gap = feature_df[["app_name", "aspect", "feature", "feature_present",
                      "promise_source", "description_claimed"]].copy()
    gap = gap.merge(per, on=["app_name", "aspect"], how="left")
    gap[["n_mentions", "n_negative"]] = gap[["n_mentions", "n_negative"]].fillna(0)

    if len(demand_df):
        dem = (
            demand_df.groupby(["app_name", "aspect"], observed=True)
            .agg(demand_signals=("reviewId", "nunique"))
            .reset_index()
        )
        gap = gap.merge(dem, on=["app_name", "aspect"], how="left")
    gap["demand_signals"] = gap.get("demand_signals", pd.Series(0, index=gap.index)).fillna(0)

    # 6a — delivery gap.
    gap["promised"] = gap["feature_present"] >= 1
    gap["broken_promise"] = (
        gap["promised"]
        & (gap["neg_rate"] > BROKEN_PROMISE_NEG_RATE)
        & (gap["n_mentions"] >= BROKEN_PROMISE_MIN_N)
    )
    broken = gap[gap["broken_promise"]].sort_values("neg_rate", ascending=False)
    log(f"6a DELIVERY GAP — {len(broken)} (app, feature) pairs flagged as broken promises")
    log(f"   criteria: promised, neg_rate > {BROKEN_PROMISE_NEG_RATE:.0%}, n >= {BROKEN_PROMISE_MIN_N}")
    for _, r in broken.head(20).iterrows():
        log(f"    {str(r['app_name'])[:30]:30s} {r['aspect']:22s} "
            f"{r['neg_rate']:.0%} neg of {r['n_mentions']:>5.0f}")

    # 6b — coverage gap, aggregated to the ecosystem.
    absent = gap[~gap["promised"]]
    unmet = (
        absent.groupby("aspect", observed=True)
        .agg(
            apps_lacking=("app_name", "nunique"),
            demand_volume=("demand_signals", "sum"),
            negative_mentions=("n_negative", "sum"),
        )
        .reset_index()
    )
    unmet["unmet_score"] = unmet["demand_volume"] * unmet["apps_lacking"]
    unmet = unmet.sort_values("unmet_score", ascending=False)

    log("")
    log("6b COVERAGE GAP — unmet needs ranked by (demand volume × apps failing to serve):")
    log(f"    {'aspect':24s} {'apps lacking':>12s} {'demand':>8s} {'score':>10s}")
    for _, r in unmet.iterrows():
        log(f"    {r['aspect']:24s} {r['apps_lacking']:>12.0f} "
            f"{r['demand_volume']:>8.0f} {r['unmet_score']:>10.0f}")

    unmet.to_csv(DATA_DIR / "unmet_needs_ranking.csv", index=False)
    gap.to_csv(OUTPUT_GAP, index=False)
    log("")
    log(f"Saved → {OUTPUT_GAP}")
    log(f"Saved → {DATA_DIR / 'unmet_needs_ranking.csv'}")
    log("→ This ranked list is the bridge to the design-implications section (§16.2).")
    return gap


# ─── §9.7–9.8 ───────────────────────────────────────────────────────────────────

def multiple_comparisons_correction(results: pd.DataFrame) -> pd.DataFrame:
    """
    §9.7 — Benjamini–Hochberg FDR.

    Not Bonferroni: with 22 features × multiple aspects it is so conservative
    that it guarantees a null result. Effect sizes with CIs are the headline;
    q-values are reported alongside.
    """
    section("§9.7  Multiple-comparisons correction (Benjamini–Hochberg)")

    if results.empty:
        return results
    if "family" not in results.columns:
        results["family"] = "primary"
    primary = results["family"] == "primary"
    estimable = results["p_value"].notna()
    valid = primary & estimable

    results["q_value"] = np.nan
    try:
        from statsmodels.stats.multitest import multipletests
        _, q, _, _ = multipletests(results.loc[valid, "p_value"], alpha=0.05, method="fdr_bh")
        results.loc[valid, "q_value"] = q
    except ImportError:
        log("statsmodels missing — skipping FDR.", level="WARN")
        return results

    results["significant_q05"] = results["q_value"] < 0.05

    # Report the full denominator. "17 of 22 survive" is uninterpretable without
    # knowing how many coefficients existed and why the rest were left out — and
    # a reviewer will read a silently shrunken family as p-hacking.
    n_total = len(results)
    n_sens = int((~primary).sum())
    n_nonest = int((primary & ~estimable).sum())
    log(f"Coefficients recorded: {n_total}")
    log(f"  in the correction family (primary, estimable): {int(valid.sum())}")
    log(f"  excluded — sensitivity/robustness refits:      {n_sens}")
    log(f"  excluded — NOT ESTIMABLE (NaN p, separation):  {n_nonest}")
    if n_nonest:
        for _, r in results[primary & ~estimable].iterrows():
            log(f"      {r['model']} / {r['term']}", level="WARN")
        log("  Report these as not-estimable in the paper — NOT as null results.",
            level="WARN")
    log(f"Survive BH-FDR q < 0.05: {int(results['significant_q05'].sum())} "
        f"of {int(valid.sum())}")
    log("Report effect sizes with CIs as the headline; q-values alongside (§9.7).")
    return results


def developer_response_analysis(design: pd.DataFrame) -> pd.DataFrame:
    """
    §9.8 — Developer response analysis.

    Reply-vs-rating is CORRELATIONAL and reply targeting is non-random —
    developers reply selectively to negative reviews. No causal claim.
    """
    section("§9.8  Developer response analysis")

    rows = []
    for app, grp in design.groupby("app_name", observed=True):
        replied = grp[grp["has_reply"]]
        rows.append({
            "app_name": app,
            "n_reviews": len(grp),
            "reply_rate": grp["has_reply"].mean(),
            "median_days_to_reply": grp["days_to_reply"].median(),
            "mean_score_replied": replied["score"].astype(float).mean() if len(replied) else np.nan,
            "mean_score_unreplied": grp.loc[~grp["has_reply"], "score"].astype(float).mean(),
            "mean_score": grp["score"].astype(float).mean(),
        })

    out = pd.DataFrame(rows).sort_values("reply_rate", ascending=False)
    log(f"{'app':38s} {'reply%':>7s} {'median d':>9s} {'★replied':>9s} {'★not':>7s}")
    for _, r in out.iterrows():
        md = f"{r['median_days_to_reply']:.1f}" if pd.notna(r["median_days_to_reply"]) else "  n/a"
        log(f"    {str(r['app_name'])[:36]:36s} {r['reply_rate']:>7.1%} {md:>9s} "
            f"{r['mean_score_replied']:>9.2f} {r['mean_score_unreplied']:>7.2f}")

    log("")
    log("CAVEAT for the paper: developers reply selectively to negative reviews, so "
        "the replied/unreplied star gap is confounded by selection. Correlational only.")
    return out


# ─── Pipeline ───────────────────────────────────────────────────────────────────

def main() -> None:
    p = base_parser(__doc__ or "Phase 5 — statistical modelling")
    p.add_argument("--models", default="all",
                   help="Comma-separated subset of m1,m2,m3,m4,m5,m6 (default: all).")
    args = p.parse_args()
    set_seed(args.seed)

    require(SENTIMENT_PATH, "python src/scripts/02b_sentiment_roberta.py")
    require(ASPECT_PATH, "python src/scripts/03_absa.py")
    require(FEATURE_MATRIX_PATH, "python src/scripts/03b_promise_extraction.py")

    df = pd.read_parquet(SENTIMENT_PATH)
    df = df[df["corpus_text"]].copy()
    aspect_df = pd.read_parquet(ASPECT_PATH)
    feature_df = pd.read_parquet(FEATURE_MATRIX_PATH)
    demand_df = pd.read_parquet(DEMAND_PATH) if DEMAND_PATH.exists() else pd.DataFrame()
    if demand_df.empty:
        log("demand.parquet not found — RQ3/RQ4/RQ6b demand sections will be thin.", level="WARN")

    log(f"reviews={len(df):,}  aspects={len(aspect_df):,}  "
        f"features={len(feature_df):,}  demand={len(demand_df):,}")

    design = build_review_design(df, aspect_df)
    wanted = {"m1", "m2", "m3", "m4", "m5", "m6"} if args.models == "all" \
        else {m.strip().lower() for m in args.models.split(",")}

    if "m1" in wanted:
        model_m1_rq2_competing_predictors(design)
    if "m2" in wanted:
        model_m2_rq1_gamification_valence(design, aspect_df, feature_df)
    if "m3" in wanted:
        model_m3_rq4_inclusion(design, aspect_df, feature_df, demand_df)
    if "m4" in wanted:
        model_m4_rq5_bloat_hardware(design, feature_df, demand_df)
    if "m5" in wanted:
        model_m5_rq3_competitive_edge(design, aspect_df, demand_df, feature_df)
    if "m6" in wanted:
        model_m6_rq6_gap_matrix(feature_df, aspect_df, demand_df)

    results = multiple_comparisons_correction(pd.DataFrame(_RESULTS))
    dev = developer_response_analysis(design)

    section("Writing output")
    app_comparison = (
        design.groupby("app_name", observed=True)
        .agg(
            n_reviews=("reviewId", "size"),
            mean_score=("score", "mean"),
            mean_sentiment=("sent_num", "mean"),
            pct_negative=("roberta_label", lambda s: (s == "Negative").mean()),
            median_words=("n_words", "median"),
            realInstalls=("realInstalls", "first"),
            adSupported=("adSupported", "first"),
        )
        .reset_index()
        .merge(
            feature_df.groupby("app_name", observed=True)["feature_present"]
            .sum().rename("feature_count").reset_index(),
            on="app_name", how="left",
        )
        .merge(dev[["app_name", "reply_rate", "median_days_to_reply"]], on="app_name", how="left")
    )
    app_comparison.to_csv(OUTPUT_COMPARISON, index=False)
    results.to_csv(OUTPUT_RESULTS, index=False)
    log(f"Saved → {OUTPUT_COMPARISON}")
    log(f"Saved → {OUTPUT_RESULTS}")
    log("")
    log("Next: the notebooks/07_rq*.py deep dives, then 08_visualizations.py")


if __name__ == "__main__":
    main()
