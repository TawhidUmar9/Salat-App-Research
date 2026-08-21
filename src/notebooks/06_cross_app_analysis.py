# ---
# jupyter:
#   jupytext:
#     text_representation: {extension: .py, format_name: percent}
# ---

# %% [markdown]
# # Phase 5 — Cross-App Analysis
#
# Ecosystem-level descriptives and the model-results review pass. This is the
# notebook to read *before* the RQ deep dives: it establishes what the corpus
# looks like and whether the models behaved.
#
# See `implementation_plan.md` §9.

# %%
from _nbinit import *  # noqa: F403

plt, WONG = setup_plots()  # noqa: F405

reviews = load_reviews()      # noqa: F405
aspects = load_aspects()      # noqa: F405
features = load_features()    # noqa: F405
demand = load_demand()        # noqa: F405
results = load_model_results()  # noqa: F405
comparison = pd.read_csv(DATA_DIR / "app_comparison.csv")  # noqa: F405,F405

# %% [markdown]
# ## 1. App comparison table
#
# This is the table the Methods section describes the corpus with.

# %%
display(comparison.sort_values("n_reviews", ascending=False).round(3))  # noqa: F821

# %% [markdown]
# ### The ceiling-compression problem (§2.1)
#
# App-level star means sit in a narrow band near the top of a bounded scale.
# This is *why* the analysis moved to the review level.

# %%
s = comparison["mean_score"].dropna()
print(f"App-level mean star: min={s.min():.2f}  max={s.max():.2f}  "
      f"range={s.max()-s.min():.2f}  sd={s.std():.3f}")
print(f"Review-level star sd: {reviews['score'].astype(float).std():.3f}")
print("\n→ Between-app variance is small relative to within-app variance. "
      "A 26-unit regression on this DV is uninterpretable (§2.1).")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].hist(s, bins=15, color=WONG[0], edgecolor="white")
axes[0].set_xlabel("App mean star"); axes[0].set_ylabel("Apps")
axes[0].set_title("App-level means — compressed")
axes[1].hist(reviews["score"].dropna().astype(float), bins=5, color=WONG[1], edgecolor="white")
axes[1].set_xlabel("Review star"); axes[1].set_ylabel("Reviews")
axes[1].set_title("Review-level — the actual variance")
fig.tight_layout()

# %% [markdown]
# ## 2. Aspect coverage across the ecosystem

# %%
cov = aspects.groupby("aspect").agg(
    mentions=("reviewId", "nunique"),
    apps=("app_name", "nunique"),
    pct_negative=("sentiment_label", lambda s: (s == "Negative").mean()),
    lexicon_share=("match_method", lambda s: (s == "lexicon").mean()),
).sort_values("mentions", ascending=False)
cov["mention_rate"] = cov["mentions"] / len(reviews)
display(cov.round(4))  # noqa: F821

# %%
d = cov.sort_values("mentions")
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(len(d)), d["mentions"],
        color=[WONG[4] if r > 0.4 else WONG[0] for r in d["pct_negative"]])
ax.set_yticks(range(len(d))); ax.set_yticklabels(d.index, fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("Reviews mentioning (log)")
ax.set_title("Aspect mention volume  (orange = >40% negative)")
for i, r in enumerate(d["pct_negative"]):
    ax.annotate(f"{r:.0%}", (d["mentions"].iloc[i], i), xytext=(4, 0),
                textcoords="offset points", va="center", fontsize=7, color="grey")
fig.tight_layout()

# %% [markdown]
# ## 3. Model results review
#
# Check convergence and effect magnitudes before writing anything up.

# %%
if len(results):
    display(results.round(5))  # noqa: F821
    print(f"\nTotal coefficients: {len(results)}")
    print(f"Surviving BH q < 0.05: {int(results['significant_q05'].sum())}")
    print("\nBy model:")
    display(results.groupby("model").agg(  # noqa: F821
        n_terms=("term", "size"),
        n_significant=("significant_q05", "sum"),
        max_abs_effect=("estimate", lambda x: x.abs().max()),
    ).round(4))
else:
    print("model_results.csv not found — run 06_models.py")

# %% [markdown]
# ### Effect sizes are the headline, not p-values (§9.7)

# %%
if len(results):
    # Rank by |effect| so a large negative coefficient is as visible as a positive one.
    withci = results.dropna(subset=["ci_low", "ci_high"])
    d = withci.reindex(withci["estimate"].abs().sort_values().index).tail(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    y = range(len(d))
    ax.errorbar(d["estimate"], y,
                xerr=[d["estimate"] - d["ci_low"], d["ci_high"] - d["estimate"]],
                fmt="none", ecolor="grey", lw=1.2, capsize=3)
    ax.scatter(d["estimate"], y, s=50,
               c=[WONG[2] if sig else WONG[4] for sig in d["significant_q05"].fillna(False)],
               zorder=3)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{r.model}: {str(r.term)[:26]}" for r in d.itertuples()], fontsize=7)
    ax.set_xlabel("Estimate (95% CI)  —  green survives BH q<0.05")
    ax.set_title("Largest effects across M1–M6")
    fig.tight_layout()

# %% [markdown]
# ## 4. Feature matrix and promise-source agreement

# %%
counts = features.groupby("app_name").agg(
    feature_count=("feature_present", "sum"),
    source=("promise_source", "first"),
    csv_available=("csv_annotated", lambda s: s.notna().any()),
).sort_values("feature_count", ascending=False)
display(counts)  # noqa: F821

agree_path = DATA_DIR / "promise_source_agreement.csv"  # noqa: F405
if agree_path.exists():
    ag = pd.read_csv(agree_path)  # noqa: F405
    print(f"\nMean Cohen's κ (CSV vs description): {ag['cohens_kappa'].mean(skipna=True):.3f}")
    display(ag.sort_values("cohens_kappa").round(3))  # noqa: F821
    print("\nFeatures where description extraction is unreliable (κ < 0.4) should be "
          "reported with the CSV as primary and the disagreement disclosed.")

# %% [markdown]
# ## 5. Developer response

# %%
if "reply_rate" in comparison.columns:
    d = comparison.dropna(subset=["reply_rate", "mean_score"]).sort_values("reply_rate")
    display(d[["app_name", "n_reviews", "reply_rate",  # noqa: F821
               "median_days_to_reply", "mean_score"]].round(3))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(d["reply_rate"], d["mean_score"],
               s=40 + 250 * (np.log1p(d["n_reviews"]) / np.log1p(d["n_reviews"]).max()),  # noqa: F405
               color=WONG[0], alpha=0.85, edgecolor="white")
    for _, r in d.iterrows():
        ax.annotate(str(r["app_name"])[:16], (r["reply_rate"], r["mean_score"]),
                    fontsize=6, xytext=(3, 3), textcoords="offset points", alpha=0.75)
    ax.set_xlabel("Developer reply rate"); ax.set_ylabel("Mean star")
    ax.set_title("Reply rate vs. rating — correlational only (§9.8)")
    fig.tight_layout()

    print("\nCAVEAT: developers reply selectively to negative reviews. "
          "The replied/unreplied gap is confounded by selection — no causal claim.")

# %% [markdown]
# ## 6. Corpus caveats to restate in the paper (§2.4)

# %%
print(f"Total reviews analysed (C_text): {len(reviews):,}")
print(f"Languages: {reviews['lang'].nunique()}; "
      f"non-English share {(reviews['lang'] != 'en').mean():.1%}")
print(f"Median review length: {reviews['n_words'].median():.0f} words")
print()
print("State in the Methods section:")
print("  1. Sort.NEWEST → recency-weighted, not a uniform all-time sample.")
print("  2. Text reviews only — ratings-without-text are invisible here.")
print("  3. lang=en/country=us sets the storefront, NOT the review language.")
print("  4. Non-English sentiment is scored natively (XLM-R), reported separately, "
      "never pooled into headline numbers (§16.3).")
