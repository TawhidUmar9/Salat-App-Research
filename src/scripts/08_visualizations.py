"""
Phase 7 — Visualization
========================
Paper-ready figures: matplotlib + seaborn with shared style.

Input:
    - All data/ outputs from previous phases

Output:
    - figures/*.png  (300 DPI)
    - figures/*.pdf  (vector)

Style: figures/style.mplstyle — colorblind-safe palette,
consistent app color mapping across every figure.

See implementation_plan.md §11 for full specification.

Figures whose inputs are missing are skipped with a logged reason rather than
crashing the run, so this can be re-run as upstream stages complete.

Usage:
    python src/scripts/08_visualizations.py
    python src/scripts/08_visualizations.py --only 5,6,7
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    ASPECT_PATH, DATA_DIR, DEMAND_PATH, FEATURE_MATRIX_PATH, FIGURES_DIR,
    SENTIMENT_PATH, TEMPORAL_PATH,
    base_parser, log, section, set_seed,
)

STYLE_PATH = FIGURES_DIR / "style.mplstyle"

#: Wong (2011) colorblind-safe palette.
WONG = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
        "#D55E00", "#56B4E9", "#F0E442", "#000000"]

#: Semantic colors reused across every sentiment figure.
SENTIMENT_COLORS = {"Positive": "#009E73", "Neutral": "#999999", "Negative": "#D55E00"}

_APP_COLORS: dict[str, str] = {}


def setup_style() -> None:
    """Load the shared style and build a stable app→color mapping."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))
        log(f"Loaded style from {STYLE_PATH}")
    else:
        log("style.mplstyle not found — using matplotlib defaults.", level="WARN")

    if SENTIMENT_PATH.exists():
        apps = sorted(pd.read_parquet(SENTIMENT_PATH, columns=["app_name"])["app_name"].dropna().unique())
        # Wong first (maximally distinguishable), then tab20 for the long tail.
        # Sorted app order keeps the mapping identical across every figure.
        from matplotlib.colors import to_hex
        cmap = plt.get_cmap("tab20")
        for i, app in enumerate(apps):
            _APP_COLORS[app] = WONG[i] if i < len(WONG) else to_hex(cmap((i - len(WONG)) % cmap.N))
        log(f"App color map fixed for {len(apps)} apps.")


def app_color(app: str):
    return _APP_COLORS.get(app, "#666666")


def save_figure(fig, name: str) -> None:
    """Write PNG @300 DPI and vector PDF."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
    log(f"    saved figures/{name}.png + .pdf")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _load(path: Path, label: str):
    if not path.exists():
        log(f"  SKIP — {label} not found ({path.name})", level="WARN")
        return None
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


# ─── Figures (§11) ──────────────────────────────────────────────────────────────

def fig01_corpus_composition():
    """Fig 1 — Corpus composition: reviews/app, language mix, length distribution."""
    import matplotlib.pyplot as plt
    df = _load(SENTIMENT_PATH, "master_reviews_with_sentiment")
    if df is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    counts = df["app_name"].value_counts().head(15).sort_values()
    axes[0].barh(range(len(counts)), counts.values,
                 color=[app_color(a) for a in counts.index])
    axes[0].set_yticks(range(len(counts)))
    axes[0].set_yticklabels([str(a)[:26] for a in counts.index], fontsize=8)
    axes[0].set_xlabel("Reviews")
    axes[0].set_title("Reviews per app (top 15)")
    axes[0].set_xscale("log")

    langs = df["lang"].value_counts().head(8)
    axes[1].bar(range(len(langs)), langs.values, color=WONG[0])
    axes[1].set_xticks(range(len(langs)))
    axes[1].set_xticklabels(langs.index, rotation=45)
    axes[1].set_ylabel("Reviews")
    axes[1].set_title("Language mix")
    axes[1].set_yscale("log")

    axes[2].hist(df["n_words"].clip(upper=60), bins=60, color=WONG[1], edgecolor="none")
    axes[2].axvline(5, color=WONG[4], ls="--", lw=1.5, label="C_text threshold (5 words)")
    axes[2].set_xlabel("Words per review (clipped at 60)")
    axes[2].set_ylabel("Reviews")
    axes[2].set_title("Review length")
    axes[2].legend()

    fig.suptitle("Corpus composition", fontsize=13)
    fig.tight_layout()
    save_figure(fig, "fig01_corpus_composition")


def fig02_sentiment_agreement():
    """Fig 2 — Sentiment method agreement (VADER / RoBERTa / star) + κ."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    df = _load(SENTIMENT_PATH, "master_reviews_with_sentiment")
    if df is None:
        return
    d = df.dropna(subset=["vader_label", "roberta_label", "star_label"])
    if d.empty:
        log("  SKIP — no rows with all three labels.", level="WARN")
        return

    labels = ["Negative", "Neutral", "Positive"]
    pairs = [("vader_label", "roberta_label"), ("vader_label", "star_label"),
             ("roberta_label", "star_label")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (a, b) in zip(axes, pairs):
        cm = confusion_matrix(d[a], d[b], labels=labels, normalize="true")
        kappa = cohen_kappa_score(d[a], d[b], labels=labels)
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xticks(range(3)); ax.set_xticklabels(labels, rotation=45)
        ax.set_yticks(range(3)); ax.set_yticklabels(labels)
        ax.set_xlabel(b.replace("_label", "")); ax.set_ylabel(a.replace("_label", ""))
        ax.set_title(f"κ = {kappa:.3f}")
        ax.grid(False)

    fig.colorbar(im, ax=axes, shrink=0.8, label="Row-normalized share")
    fig.suptitle(f"Sentiment method agreement (n = {len(d):,})", fontsize=13)
    save_figure(fig, "fig02_sentiment_agreement")


def fig03_promise_source_agreement():
    """Fig 3 — Promise-source agreement: CSV vs. store description."""
    import matplotlib.pyplot as plt
    rep = _load(DATA_DIR / "promise_source_agreement.csv", "promise_source_agreement")
    if rep is None:
        return

    rep = rep.sort_values("cohens_kappa", na_position="first")
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = [WONG[2] if k > 0.6 else WONG[1] if k > 0.4 else WONG[4]
              for k in rep["cohens_kappa"].fillna(-1)]
    ax.barh(range(len(rep)), rep["cohens_kappa"].fillna(0), color=colors)
    ax.set_yticks(range(len(rep)))
    ax.set_yticklabels([str(f)[:34] for f in rep["feature"]], fontsize=8)
    ax.set_xlabel("Cohen's κ  (hand annotation vs. store description)")
    for x, lab in [(0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")]:
        ax.axvline(x, color="grey", ls=":", lw=1)
        ax.text(x, len(rep) - 0.4, lab, fontsize=7, ha="center", color="grey")
    ax.set_title("Promise-source agreement per feature")
    fig.tight_layout()
    save_figure(fig, "fig03_promise_source_agreement")


def fig04_aspect_sentiment_heatmap():
    """Fig 4 — Aspect × sentiment heatmap, all apps."""
    import matplotlib.pyplot as plt
    asp = _load(ASPECT_PATH, "aspect_sentiments")
    if asp is None:
        return

    d = asp[~asp["is_request"].fillna(False)]
    piv = d.pivot_table(index="app_name", columns="aspect", values="sentiment_label",
                        aggfunc=lambda s: (s == "Negative").mean())
    counts = d.pivot_table(index="app_name", columns="aspect", values="reviewId", aggfunc="count")
    piv = piv.where(counts >= 10)          # suppress cells too thin to read

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(piv.values, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([str(a)[:30] for a in piv.index], fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.0%}"[:-1], ha="center", va="center", fontsize=5.5,
                        color="white" if v > 0.6 or v < 0.15 else "black")
    fig.colorbar(im, ax=ax, label="Negative-sentiment rate", shrink=0.7)
    ax.set_title("Aspect × app negative-sentiment rate (cells with n ≥ 10)")
    ax.grid(False)
    fig.tight_layout()
    save_figure(fig, "fig04_aspect_sentiment_heatmap")


def fig05_rq2_complaint_effects():
    """Fig 5 — RQ2: complaint-type effect on star rating (forest plot)."""
    import matplotlib.pyplot as plt
    res = _load(DATA_DIR / "model_results.csv", "model_results")
    if res is None:
        return

    m1 = res[(res["model"] == "M1") & res["term"].str.startswith("complaint_", na=False)]
    if m1.empty:
        log("  SKIP — no M1 complaint coefficients.", level="WARN")
        return
    m1 = m1.sort_values("estimate")

    accuracy = {"prayer_times_accuracy", "qibla", "madhab", "calc_method"}
    colors = [WONG[4] if t.replace("complaint_", "") in accuracy else WONG[0]
              for t in m1["term"]]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = range(len(m1))
    ax.errorbar(m1["estimate"], y,
                xerr=[m1["estimate"] - m1["ci_low"], m1["ci_high"] - m1["estimate"]],
                fmt="none", ecolor="grey", lw=1.2, capsize=3)
    ax.scatter(m1["estimate"], y, c=colors, s=55, zorder=3)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels([t.replace("complaint_", "") for t in m1["term"]], fontsize=9)
    ax.set_xlabel("Effect on star rating (stars, 95% CI)")
    ax.set_title("RQ2 — which complaints cost the most stars?")

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=WONG[4], label="Accuracy family"),
        Line2D([], [], marker="o", ls="", color=WONG[0], label="Comparison (UI, bugs, ads…)"),
    ], loc="lower right")
    fig.tight_layout()
    save_figure(fig, "fig05_rq2_complaint_effects")


def fig06_rq1_tracker_sentiment():
    """Fig 6 — RQ1: tracker sentiment distribution by tracker tier."""
    import matplotlib.pyplot as plt
    asp = _load(ASPECT_PATH, "aspect_sentiments")
    feat = _load(FEATURE_MATRIX_PATH, "feature_matrix_combined")
    if asp is None or feat is None:
        return

    tracker = asp[asp["aspect"].isin(["prayer_tracker", "tracker_score", "goal_system"])]
    tracker = tracker[~tracker["is_request"].fillna(False)].dropna(subset=["prob_pos", "prob_neg"])
    if tracker.empty:
        log("  SKIP — no tracker-aspect rows.", level="WARN")
        return

    tier = {}
    for app, grp in feat.groupby("app_name", observed=True):
        has = dict(zip(grp["aspect"], grp["feature_present"].fillna(0)))
        tier[app] = ("tracker + score" if has.get("tracker_score", 0) >= 1
                     else "tracker only" if has.get("prayer_tracker", 0) >= 1 else "none")
    tracker = tracker.assign(
        tier=tracker["app_name"].map(tier).fillna("none"),
        valence=tracker["prob_pos"] - tracker["prob_neg"],
    )

    order = ["none", "tracker only", "tracker + score"]
    data = [tracker.loc[tracker["tier"] == t, "valence"].to_numpy() for t in order]
    data = [d for d in data if len(d) > 10]
    labels = [t for t, d in zip(order, data) if len(d) > 10]

    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(WONG[i % len(WONG)])
        body.set_alpha(0.65)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_ylabel("Sentiment valence  (P(pos) − P(neg))")
    ax.set_title("RQ1 — tracker sentiment by tracker tier\n"
                 "A bimodal shape, not the mean, is the love/hate signal", fontsize=11)
    fig.tight_layout()
    save_figure(fig, "fig06_rq1_tracker_sentiment")


def fig07_rq1_guilt_vs_motivation():
    """Fig 7 — RQ1: guilt vs. motivation lexicon rates (diverging bar)."""
    import matplotlib.pyplot as plt
    asp = _load(ASPECT_PATH, "aspect_sentiments")
    if asp is None:
        return

    affect = asp[asp["aspect"] == "spiritual_affect"].copy()
    if affect.empty:
        log("  SKIP — no spiritual_affect rows.", level="WARN")
        return

    guilt_re = r"\b(guilt|guilty|anxious|anxiety|pressure|ashamed|shame|stress|fail)"
    motiv_re = r"\b(motivat|encourag|discipline|consisten|barakah|helpful|improve)"
    text = affect["triggering_sentence"].fillna("").str.lower()
    affect["guilt"] = text.str.contains(guilt_re, regex=True)
    affect["motivation"] = text.str.contains(motiv_re, regex=True)

    per = affect.groupby("app_name", observed=True)[["guilt", "motivation"]].mean()
    per = per[(per.sum(axis=1) > 0)].sort_values("guilt")
    if per.empty:
        log("  SKIP — no guilt/motivation matches.", level="WARN")
        return

    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(per))))
    y = np.arange(len(per))
    ax.barh(y, -per["guilt"], color=WONG[4], label="Guilt / anxiety")
    ax.barh(y, per["motivation"], color=WONG[2], label="Motivation / discipline")
    ax.set_yticks(y)
    ax.set_yticklabels([str(a)[:30] for a in per.index], fontsize=8)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Share of spiritual-affect sentences")
    ax.set_title("RQ1 — guilt vs. motivation framing per app")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, "fig07_rq1_guilt_vs_motivation")


def fig08_rq5_bloat_curve():
    """Fig 8 — RQ5: feature count vs. complexity-complaint rate."""
    import matplotlib.pyplot as plt
    comp = _load(DATA_DIR / "app_comparison.csv", "app_comparison")
    asp = _load(ASPECT_PATH, "aspect_sentiments")
    if comp is None or asp is None:
        return

    bloat = (
        asp[asp["aspect"] == "complexity_bloat"]
        .groupby("app_name", observed=True)["reviewId"].nunique().rename("bloat_mentions")
    )
    d = comp.merge(bloat, on="app_name", how="left")
    d["bloat_mentions"] = d["bloat_mentions"].fillna(0)
    d["bloat_rate"] = d["bloat_mentions"] / d["n_reviews"]
    d = d.dropna(subset=["feature_count"])
    if len(d) < 4:
        log("  SKIP — too few apps with a feature count.", level="WARN")
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    sizes = 40 + 260 * (np.log1p(d["n_reviews"]) / np.log1p(d["n_reviews"]).max())
    ax.scatter(d["feature_count"], d["bloat_rate"], s=sizes,
               c=[app_color(a) for a in d["app_name"]], alpha=0.85, edgecolor="white", lw=0.8)

    if len(d) >= 5:
        x, y = d["feature_count"].to_numpy(float), d["bloat_rate"].to_numpy(float)
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, np.polyval(coef, xs), color=WONG[0], lw=2)
        # Bootstrap band — n is small, so show the uncertainty honestly.
        rng = np.random.default_rng(0)
        boots = np.array([
            np.polyval(np.polyfit(*(lambda i: (x[i], y[i]))(rng.integers(0, len(x), len(x))), 1), xs)
            for _ in range(400)
        ])
        ax.fill_between(xs, np.percentile(boots, 2.5, axis=0),
                        np.percentile(boots, 97.5, axis=0), color=WONG[0], alpha=0.15)

    for _, r in d.iterrows():
        ax.annotate(str(r["app_name"])[:16], (r["feature_count"], r["bloat_rate"]),
                    fontsize=6, alpha=0.75, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Feature count (0–20)")
    ax.set_ylabel("Complexity-complaint rate")
    ax.set_title(f"RQ5 — feature bloat vs. complexity complaints (n = {len(d)} apps)\n"
                 "Exploratory: app-level n is small; marker size = log review volume", fontsize=11)
    fig.tight_layout()
    save_figure(fig, "fig08_rq5_bloat_curve")


def fig09_rq4_women_aspect():
    """Fig 9 — RQ4: women-aspect mention volume & sentiment per app."""
    import matplotlib.pyplot as plt
    asp = _load(ASPECT_PATH, "aspect_sentiments")
    if asp is None:
        return

    w = asp[asp["aspect"] == "women_period"]
    if w.empty:
        log("  SKIP — no women_period mentions. Report that scarcity as a finding (§9.3).",
            level="WARN")
        return

    per = w.groupby("app_name", observed=True).agg(
        mentions=("reviewId", "nunique"),
        neg_rate=("sentiment_label", lambda s: (s == "Negative").mean()),
    ).sort_values("mentions", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(per))))
    y = np.arange(len(per))
    ax.barh(y, per["mentions"], color=[app_color(a) for a in per.index])
    ax.set_yticks(y); ax.set_yticklabels([str(a)[:30] for a in per.index], fontsize=8)
    ax.set_xlabel("Reviews mentioning menstruation / women's exemptions")

    ax2 = ax.twiny()
    ax2.plot(per["neg_rate"], y, "o", color=WONG[4], ms=6)
    ax2.set_xlim(0, 1); ax2.set_xlabel("Negative-sentiment rate", color=WONG[4])
    ax2.tick_params(axis="x", colors=WONG[4]); ax2.grid(False)

    ax.set_title(f"RQ4 — women-aspect volume and sentiment (total n = {per['mentions'].sum():,})")
    fig.tight_layout()
    save_figure(fig, "fig09_rq4_women_aspect")


def fig10_rq6a_delivery_gap():
    """Fig 10 — RQ6a: delivery gap, app × feature."""
    import matplotlib.pyplot as plt
    gap = _load(DATA_DIR / "gap_matrix.csv", "gap_matrix")
    if gap is None:
        return

    promised = gap[gap["promised"].fillna(False)]
    if promised.empty:
        log("  SKIP — no promised features.", level="WARN")
        return
    piv = promised.pivot_table(index="app_name", columns="aspect", values="neg_rate")
    n = promised.pivot_table(index="app_name", columns="aspect", values="n_mentions")
    piv = piv.where(n >= 10)

    fig, ax = plt.subplots(figsize=(13, 8))
    im = ax.imshow(piv.values, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([str(a)[:30] for a in piv.index], fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if pd.notna(v) and v > 0.40:
                ax.text(j, i, "✕", ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(im, ax=ax, label="Negative-sentiment rate on a promised feature", shrink=0.7)
    ax.set_title("RQ6a — delivery gap: ✕ marks a broken promise (>40% negative, n ≥ 30)")
    ax.grid(False)
    fig.tight_layout()
    save_figure(fig, "fig10_rq6a_delivery_gap")


def fig11_rq6b_unmet_needs():
    """Fig 11 — RQ6b: unmet-need ranking (demand × apps failing)."""
    import matplotlib.pyplot as plt
    unmet = _load(DATA_DIR / "unmet_needs_ranking.csv", "unmet_needs_ranking")
    if unmet is None:
        return

    d = unmet.sort_values("unmet_score").tail(15)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(d))))
    y = np.arange(len(d))
    ax.hlines(y, 0, d["unmet_score"], color="grey", lw=1.2)
    ax.scatter(d["unmet_score"], y, s=70, color=WONG[4], zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(d["aspect"], fontsize=9)
    ax.set_xlabel("Unmet-need score  (demand volume × apps lacking the feature)")
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.annotate(f"{r['demand_volume']:.0f} req · {r['apps_lacking']:.0f} apps",
                    (r["unmet_score"], yi), fontsize=7, xytext=(6, 0),
                    textcoords="offset points", va="center", color="grey")
    ax.set_title("RQ6b — unmet needs across the ecosystem\n"
                 "The ranked bridge to design implications", fontsize=11)
    fig.tight_layout()
    save_figure(fig, "fig11_rq6b_unmet_needs")


def fig12_temporal_aspect_volume():
    """Fig 12 — Temporal: aspect volume over time (small multiples)."""
    import matplotlib.pyplot as plt
    vol = _load(DATA_DIR / "aspect_volume_over_time.csv", "aspect_volume_over_time")
    if vol is None:
        return

    vol["quarter"] = pd.to_datetime(vol["quarter"])
    focus = ["ads_intrusive", "privacy_data", "tracker_score", "prayer_tracker",
             "monetization", "stability_bugs", "qibla", "women_period", "ui_design"]
    focus = [a for a in focus if a in set(vol["aspect"])]
    if not focus:
        log("  SKIP — none of the focus aspects present.", level="WARN")
        return

    ncol = 3
    nrow = int(np.ceil(len(focus) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.1 * nrow), sharex=True)
    for ax, aspect in zip(np.ravel(axes), focus):
        d = vol[vol["aspect"] == aspect].sort_values("quarter")
        ax.plot(d["quarter"], d["mention_rate"], color=WONG[0], lw=1.8)
        ax.fill_between(d["quarter"], 0, d["mention_rate"], color=WONG[0], alpha=0.15)
        ax2 = ax.twinx()
        ax2.bar(d["quarter"], d["total_reviews"], width=70, color="grey", alpha=0.18)
        ax2.set_yticks([]); ax2.grid(False)
        ax.set_title(aspect, fontsize=10)
        ax.set_ylabel("mention rate", fontsize=8)
    for ax in np.ravel(axes)[len(focus):]:
        ax.set_visible(False)
    fig.suptitle("Aspect mention rate over time (grey bars = review volume; "
                 "early periods are thin — §2.4)", fontsize=11)
    fig.tight_layout()
    save_figure(fig, "fig12_temporal_aspect_volume")


def fig13_muslim_pro_privacy():
    """Fig 13 — Muslim Pro privacy changepoint (interrupted time series)."""
    import matplotlib.pyplot as plt
    its = _load(DATA_DIR / "muslim_pro_privacy_its.csv", "muslim_pro_privacy_its")
    if its is None:
        return

    its["month"] = pd.to_datetime(its["month"])
    event = pd.Timestamp("2020-11-16")
    d = its[its["months_from_event"].between(-24, 24)]
    if d.empty:
        log("  SKIP — no data in the ±24-month window.", level="WARN")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, col, label, color in [
        (axes[0], "privacy_rate", "Privacy mention rate", WONG[4]),
        (axes[1], "mean_score", "Mean star rating", WONG[0]),
    ]:
        for period, style in [("pre", "-"), ("post", "-")]:
            sub = d[d["period"] == period]
            ax.plot(sub["month"], sub[col], style, color=color, lw=1.8, marker="o", ms=3)
        ax.axvline(event, color="black", ls="--", lw=1.5)
        ax.set_ylabel(label)
    axes[0].annotate("Nov 2020: data-sale reporting", (event, axes[0].get_ylim()[1]),
                     xytext=(8, -12), textcoords="offset points", fontsize=9)
    axes[0].set_title("Muslim Pro — interrupted time series around the 2020 privacy reporting\n"
                      "Correlational: other concurrent changes are not controlled", fontsize=11)
    fig.tight_layout()
    save_figure(fig, "fig13_muslim_pro_privacy")


def fig14_dev_reply_rating():
    """Fig 14 — Developer reply rate vs. rating trajectory."""
    import matplotlib.pyplot as plt
    comp = _load(DATA_DIR / "app_comparison.csv", "app_comparison")
    if comp is None or "reply_rate" not in comp.columns:
        log("  SKIP — app_comparison lacks reply_rate.", level="WARN")
        return

    d = comp.dropna(subset=["reply_rate", "mean_score"])
    if len(d) < 4:
        log("  SKIP — too few apps.", level="WARN")
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    sizes = 40 + 260 * (np.log1p(d["n_reviews"]) / np.log1p(d["n_reviews"]).max())
    ax.scatter(d["reply_rate"], d["mean_score"], s=sizes,
               c=[app_color(a) for a in d["app_name"]], alpha=0.85, edgecolor="white", lw=0.8)
    if len(d) >= 5:
        x, y = d["reply_rate"].to_numpy(float), d["mean_score"].to_numpy(float)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, np.polyval(np.polyfit(x, y, 1), xs), color=WONG[0], lw=2, alpha=0.8)
    for _, r in d.iterrows():
        ax.annotate(str(r["app_name"])[:16], (r["reply_rate"], r["mean_score"]),
                    fontsize=6, alpha=0.75, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Developer reply rate")
    ax.set_ylabel("Mean star rating")
    ax.set_title("Developer responsiveness vs. rating\n"
                 "Correlational only — reply targeting is non-random (§9.8)", fontsize=11)
    fig.tight_layout()
    save_figure(fig, "fig14_dev_reply_rating")


FIGURES = {
    1: fig01_corpus_composition, 2: fig02_sentiment_agreement,
    3: fig03_promise_source_agreement, 4: fig04_aspect_sentiment_heatmap,
    5: fig05_rq2_complaint_effects, 6: fig06_rq1_tracker_sentiment,
    7: fig07_rq1_guilt_vs_motivation, 8: fig08_rq5_bloat_curve,
    9: fig09_rq4_women_aspect, 10: fig10_rq6a_delivery_gap,
    11: fig11_rq6b_unmet_needs, 12: fig12_temporal_aspect_volume,
    13: fig13_muslim_pro_privacy, 14: fig14_dev_reply_rating,
}


def main() -> None:
    p = base_parser(__doc__ or "Phase 7 — visualizations")
    p.add_argument("--only", default=None,
                   help="Comma-separated figure numbers to regenerate, e.g. 5,6,7.")
    args = p.parse_args()
    set_seed(args.seed)

    setup_style()
    wanted = (sorted(int(x) for x in args.only.split(",")) if args.only else sorted(FIGURES))

    for num in wanted:
        fn = FIGURES.get(num)
        if fn is None:
            log(f"No figure {num}.", level="WARN")
            continue
        section(f"Figure {num} — {(fn.__doc__ or '').splitlines()[0].strip()}")
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — one bad figure must not kill the run
            log(f"  FAILED: {type(exc).__name__}: {exc}", level="ERROR")

    log("")
    log(f"Figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
