# ---
# jupyter:
#   jupytext:
#     text_representation: {extension: .py, format_name: percent}
# ---

# %% [markdown]
# # RQ5 — Feature Bloat vs. Companion Hardware
#
# *Do all-in-one apps with smart rings beat simple clean apps, or does bloat
# ruin them?*
#
# Consumes M4 (§9.4). Two halves:
# - **Bloat** is moderately powered: feature count is continuous across 26 apps
#   and the outcome is measured at review level.
# - **Hardware** is a single-app case study (iQIBLA Life), reported by name
#   rather than as an anonymous treatment cell (§2.1 move 3).
#
# **Confound to name explicitly**: feature count correlates with install base
# and company size. `log(realInstalls)` is a required control and residual
# confounding is acknowledged, not modelled away.

# %%
from _nbinit import *  # noqa: F403

plt, WONG = setup_plots()  # noqa: F405

reviews = load_reviews()      # noqa: F405
aspects = load_aspects()      # noqa: F405
features = load_features()    # noqa: F405
demand = load_demand()        # noqa: F405
results = load_model_results()  # noqa: F405

# %% [markdown]
# ## 1. Feature counts across the ecosystem

# %%
counts = (features.groupby("app_name")
          .agg(feature_count=("feature_present", "sum"),
               source=("promise_source", "first"))
          .sort_values("feature_count", ascending=False))
display(counts)  # noqa: F821

app_meta = reviews.groupby("app_name").agg(
    n_reviews=("reviewId", "size"),
    mean_star=("score", "mean"),
    mean_sentiment=("sent_num", "mean"),
    realInstalls=("realInstalls", "first"),
    adSupported=("adSupported", "first"),
)
app = counts.join(app_meta)

# %% [markdown]
# ## 2. Complaint rates vs. feature count

# %%
for aspect, col in [("complexity_bloat", "bloat_rate"), ("stability_bugs", "bug_rate")]:
    m = (aspects[aspects["aspect"] == aspect]
         .groupby("app_name")["reviewId"].nunique().rename(col))
    app = app.join(m)
    app[col] = app[col].fillna(0) / app["n_reviews"]

display(app[["feature_count", "n_reviews", "bloat_rate", "bug_rate",  # noqa: F821
             "mean_star", "mean_sentiment"]].round(4))

# %%
from scipy.stats import spearmanr  # noqa: E402

d = app.dropna(subset=["feature_count"])
print(f"App-level Spearman correlations (n = {len(d)} apps) — EXPLORATORY (§9.4):\n")
for col in ["bloat_rate", "bug_rate", "mean_sentiment", "mean_star"]:
    sub = d.dropna(subset=[col])
    if len(sub) >= 5:
        rho, p = spearmanr(sub["feature_count"], sub[col])
        print(f"  feature_count vs {col:16s} ρ = {rho:+.3f}   p = {p:.4f}")

print("\nConfound check — does feature count just track install base?")
sub = d.dropna(subset=["realInstalls"])
if len(sub) >= 5:
    rho, p = spearmanr(sub["feature_count"], np.log1p(sub["realInstalls"]))  # noqa: F405
    print(f"  feature_count vs log(realInstalls)  ρ = {rho:+.3f}  p = {p:.4f}")
    print("  → If this is high, the review-level model with log_installs is the "
          "only defensible estimate.")

# %% [markdown]
# ## 3. M4 review-level model

# %%
if len(results):
    m4 = results[results["model"].isin(["M4", "M4-app"])]
    display(m4[["model", "term", "estimate", "ci_low", "ci_high",  # noqa: F821
                "p_value", "q_value", "note"]].round(5))
else:
    print("Run `python src/scripts/06_models.py --models m4`.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, col, title in [(axes[0], "bloat_rate", "Complexity complaints"),
                       (axes[1], "bug_rate", "Stability complaints")]:
    sub = d.dropna(subset=[col])
    sizes = 40 + 250 * (np.log1p(sub["n_reviews"]) / np.log1p(sub["n_reviews"]).max())  # noqa: F405
    ax.scatter(sub["feature_count"], sub[col], s=sizes, color=WONG[0],
               alpha=0.8, edgecolor="white")
    if len(sub) >= 5:
        x, y = sub["feature_count"].to_numpy(float), sub[col].to_numpy(float)
        xs = np.linspace(x.min(), x.max(), 100)  # noqa: F405
        ax.plot(xs, np.polyval(np.polyfit(x, y, 1), xs), color=WONG[4], lw=2)  # noqa: F405
    for name, r in sub.iterrows():
        ax.annotate(str(name)[:14], (r["feature_count"], r[col]), fontsize=6,
                    alpha=0.7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Feature count"); ax.set_ylabel(col); ax.set_title(title)
fig.suptitle("RQ5 — feature bloat (marker size = log review volume; exploratory)", fontsize=11)
fig.tight_layout()

# %% [markdown]
# ## 4. Minimal-app counterexamples
#
# §10 asks for the low-feature apps that do well — the strongest evidence
# against "more features is better".

# %%
minimal = d.nsmallest(5, "feature_count")[["feature_count", "n_reviews", "mean_star",
                                           "mean_sentiment", "bloat_rate"]]
maximal = d.nlargest(5, "feature_count")[["feature_count", "n_reviews", "mean_star",
                                          "mean_sentiment", "bloat_rate"]]
print("Leanest apps:"); display(minimal.round(4))     # noqa: F821
print("Fullest apps:"); display(maximal.round(4))     # noqa: F821

# %% [markdown]
# ## 5. Companion hardware — iQIBLA Life case study

# %%
hw_mask = reviews["app_name"].astype(str).str.contains("iQIBLA", case=False, na=False)
hw = reviews[hw_mask]
print(f"iQIBLA Life reviews: {len(hw):,}")
if len(hw):
    print(f"  mean star      {hw['score'].astype(float).mean():.2f}")
    print(f"  mean sentiment {hw['sent_num'].mean():+.3f}")
    print(f"  corpus mean    {reviews['sent_num'].mean():+.3f}")

hw_aspect = aspects[aspects["aspect"] == "companion_hardware"]
print(f"\ncompanion_hardware mentions ecosystem-wide: {len(hw_aspect):,} "
      f"across {hw_aspect['app_name'].nunique()} apps")
display(hw_aspect.groupby("app_name").agg(  # noqa: F821
    mentions=("reviewId", "nunique"),
    pct_negative=("sentiment_label", lambda s: (s == "Negative").mean()),
).sort_values("mentions", ascending=False).head(12).round(3))

# %%
in_hw = hw_aspect[hw_aspect["app_name"].astype(str).str.contains("iQIBLA", case=False, na=False)]
print("Hardware sentiment INSIDE the app that ships hardware:")
display(sentiment_split(in_hw["sentiment_label"]).round(3))  # noqa: F405,F821

print("\nHardware sentiment in the other 25 apps (mostly watch-support requests):")
display(sentiment_split(  # noqa: F405,F821
    hw_aspect[~hw_aspect["app_name"].astype(str).str.contains("iQIBLA", case=False, na=False)]
    ["sentiment_label"]).round(3))

if len(demand):
    hw_demand = demand[demand["aspect"] == "companion_hardware"]
    print(f"\nWearable/watch demand across the ecosystem: {len(hw_demand):,} signals "
          f"across {hw_demand['app_name'].nunique()} apps")
    for s in hw_demand["evidence_sentence"].head(8):
        print("  •", str(s)[:140])

# %% [markdown]
# ## 6. Quotes

# %%
bloat_ids = set(aspects.loc[aspects["aspect"] == "complexity_bloat", "reviewId"])
sel = pd.concat([  # noqa: F405
    reviews[reviews["reviewId"].isin(bloat_ids)].nsmallest(10, "sent_num"),
    hw.nsmallest(5, "sent_num") if len(hw) else reviews.head(0),
])
export_quotes(sel, "rq5", n=15)  # noqa: F405

# %% [markdown]
# ## Answer to RQ5
#
# - feature_count → bloat complaints: β = ___ [CI], q = ___
# - feature_count vs log(installs): ρ = ___ (confound magnitude)
# - Leanest well-rated apps: ___
# - iQIBLA hardware sentiment: ___% negative (n = ___) — **case study, N=1**
#
# → **Verdict**: _______
