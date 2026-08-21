# ---
# jupyter:
#   jupytext:
#     text_representation: {extension: .py, format_name: percent}
# ---

# %% [markdown]
# # Phase 4 — Temporal Visualizations
#
# Interactive exploration of the trends `05_temporal.py` computed.
#
# **Standing caveat (§2.4)**: the scrape used `Sort.NEWEST`, so the corpus is
# recency-weighted and early periods are thin. Every trend plot here shows `n`
# alongside the line, and sparse cells are dropped rather than smoothed over.
#
# See `implementation_plan.md` §8.

# %%
from _nbinit import *  # noqa: F403

plt, WONG = setup_plots()  # noqa: F405

reviews = load_reviews()   # noqa: F405
trends = pd.read_parquet(TEMPORAL_PATH)  # noqa: F405,F405
trends["month"] = pd.to_datetime(trends["month"])  # noqa: F405

print(f"{len(trends):,} app-month cells")
print(f"Coverage: {trends['month'].min():%Y-%m} → {trends['month'].max():%Y-%m}")
print(f"Sparse cells (n < 20): {trends['sparse'].mean():.1%}")

# %% [markdown]
# ## 1. Corpus volume over time — the sampling curve
#
# Look at this before any trend: it explains most apparent "changes" in the
# early years.

# %%
vol = trends.groupby("month")["n"].sum()
fig, ax = plt.subplots(figsize=(11, 4))
ax.fill_between(vol.index, 0, vol.values, color=WONG[0], alpha=0.4)
ax.plot(vol.index, vol.values, color=WONG[0], lw=1.5)
ax.set_ylabel("Reviews per month"); ax.set_yscale("log")
ax.set_title("Corpus volume over time (log scale) — recency-weighted by design")
fig.tight_layout()

# %% [markdown]
# ## 2. Per-app sentiment trajectories

# %%
TOP_N = 8
big = trends.groupby("app_name")["n"].sum().nlargest(TOP_N).index

fig, ax = plt.subplots(figsize=(12, 6))
for i, app in enumerate(big):
    d = trends[(trends["app_name"] == app) & (~trends["sparse"])].sort_values("month")
    ax.plot(d["month"], d["mean_sentiment_roll3"], lw=1.8,
            color=WONG[i % len(WONG)], label=str(app)[:26])
ax.axhline(0, color="black", lw=1, ls="--")
ax.set_ylabel("Mean sentiment (3-month rolling)")
ax.legend(ncol=2, fontsize=8)
ax.set_title(f"Sentiment trajectory — {TOP_N} largest apps (sparse months dropped)")
fig.tight_layout()

# %% [markdown]
# ## 3. Changepoints

# %%
cp_path = DATA_DIR / "changepoints.csv"  # noqa: F405
if cp_path.exists():
    cp = pd.read_csv(cp_path)  # noqa: F405
    cp["changepoint_month"] = pd.to_datetime(cp["changepoint_month"])  # noqa: F405
    print(f"{len(cp)} changepoints detected")
    display(cp[cp["metric"] == "mean_sentiment"].nsmallest(15, "delta").round(4))  # noqa: F821
else:
    print("changepoints.csv not found — run 05_temporal.py")

# %%
if cp_path.exists() and len(cp):
    worst = cp[cp["metric"] == "mean_sentiment"].nsmallest(1, "delta")
    if len(worst):
        app = worst.iloc[0]["app_name"]
        when = worst.iloc[0]["changepoint_month"]
        d = trends[(trends["app_name"] == app) & (~trends["sparse"])].sort_values("month")
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(d["month"], d["mean_sentiment"], color="grey", lw=0.8, alpha=0.6)
        ax.plot(d["month"], d["mean_sentiment_roll3"], color=WONG[0], lw=2)
        ax.axvline(when, color=WONG[4], ls="--", lw=2)
        ax.annotate(f"changepoint {when:%Y-%m}", (when, ax.get_ylim()[1]),
                    xytext=(6, -14), textcoords="offset points", fontsize=9)
        ax2 = ax.twinx()
        ax2.bar(d["month"], d["n"], width=20, color="grey", alpha=0.2)
        ax2.set_ylabel("n reviews", color="grey"); ax2.grid(False)
        ax.set_ylabel("Mean sentiment")
        ax.set_title(f"Largest detected sentiment drop — {app}")
        fig.tight_layout()

# %% [markdown]
# ## 4. Muslim Pro privacy — interrupted time series

# %%
its_path = DATA_DIR / "muslim_pro_privacy_its.csv"  # noqa: F405
if its_path.exists():
    its = pd.read_csv(its_path)  # noqa: F405
    its["month"] = pd.to_datetime(its["month"])  # noqa: F405
    event = pd.Timestamp("2020-11-16")  # noqa: F405
    w = its[its["months_from_event"].between(-24, 24)]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, col, label, color in [
        (axes[0], "privacy_rate", "Privacy mention rate", WONG[4]),
        (axes[1], "mean_sentiment", "Mean sentiment", WONG[0]),
        (axes[2], "mean_score", "Mean star", WONG[2]),
    ]:
        ax.plot(w["month"], w[col], color=color, lw=1.8, marker="o", ms=3)
        ax.axvline(event, color="black", ls="--", lw=1.5)
        ax.set_ylabel(label)
    axes[0].set_title("Muslim Pro — ±24 months around the Nov 2020 data-sale reporting\n"
                      "Correlational: concurrent changes are not controlled", fontsize=11)
    fig.tight_layout()

    pre = w[w["period"] == "pre"]; post = w[w["period"] == "post"]
    print(f"privacy rate   pre={pre['privacy_rate'].mean():.3%}  post={post['privacy_rate'].mean():.3%}")
    print(f"mean sentiment pre={pre['mean_sentiment'].mean():+.3f}  post={post['mean_sentiment'].mean():+.3f}")
else:
    print("muslim_pro_privacy_its.csv not found — run 05_temporal.py")

# %% [markdown]
# ## 5. Aspect volume over time
#
# RQ1-relevant: if gamification is a recent design trend, `tracker_score`
# should be dated rather than flat.

# %%
vol_path = DATA_DIR / "aspect_volume_over_time.csv"  # noqa: F405
if vol_path.exists():
    av = pd.read_csv(vol_path)  # noqa: F405
    av["quarter"] = pd.to_datetime(av["quarter"])  # noqa: F405
    focus = ["tracker_score", "prayer_tracker", "ads_intrusive", "privacy_data",
             "monetization", "women_period", "companion_hardware", "widgets", "qibla"]
    focus = [a for a in focus if a in set(av["aspect"])]

    ncol = 3
    nrow = int(np.ceil(len(focus) / ncol))  # noqa: F405
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.1 * nrow), sharex=True)
    for ax, aspect in zip(np.ravel(axes), focus):  # noqa: F405
        d = av[av["aspect"] == aspect].sort_values("quarter")
        d = d[d["total_reviews"] >= 50]
        ax.plot(d["quarter"], d["mention_rate"], color=WONG[0], lw=1.8)
        ax.fill_between(d["quarter"], 0, d["mention_rate"], color=WONG[0], alpha=0.2)
        ax.set_title(aspect, fontsize=10)
    for ax in np.ravel(axes)[len(focus):]:  # noqa: F405
        ax.set_visible(False)
    fig.suptitle("Aspect mention rate per quarter (quarters with ≥50 reviews)", fontsize=11)
    fig.tight_layout()

    print("Biggest risers (first 3 years → last 3 years):")
    yrs = av["quarter"].dt.year
    early = av[yrs <= yrs.min() + 2].groupby("aspect")["mention_rate"].mean()
    late = av[yrs >= yrs.max() - 2].groupby("aspect")["mention_rate"].mean()
    ch = (late - early).dropna().sort_values(ascending=False)
    display(ch.head(10).to_frame("change_in_mention_rate").round(4))  # noqa: F821

# %% [markdown]
# ## 6. Version-level sentiment

# %%
ver_path = DATA_DIR / "version_sentiment.csv"  # noqa: F405
if ver_path.exists():
    ver = pd.read_csv(ver_path)  # noqa: F405
    flagged = ver[ver["flagged_drop"]] if "flagged_drop" in ver else ver.head(0)
    print(f"{len(ver)} versions with enough reviews; {len(flagged)} flagged as drops")
    display(flagged.nsmallest(15, "sentiment_delta")[  # noqa: F821
        ["app_name", "version", "n", "mean_sentiment", "sentiment_delta"]].round(3))
else:
    print("version_sentiment.csv not found — appVersion may be unpopulated.")
