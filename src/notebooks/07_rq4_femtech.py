# ---
# jupyter:
#   jupytext:
#     text_representation: {extension: .py, format_name: percent}
# ---

# %% [markdown]
# # RQ4 — Bio-Spiritual Inclusion (FemTech)
#
# *Do apps accounting for menstrual exemptions get higher ratings and user loyalty?*
#
# **Power warning (§9.3)**: only 3 apps have `Women tracking`. The between-app
# comparison is descriptive only and must be labelled as such. The well-powered
# analysis is volume, sentiment, demand, and qualitative coding — and if the
# volume turns out to be small, *that scarcity is itself the finding*.
#
# See `implementation_plan.md` §10, §9.3.

# %%
from _nbinit import *  # noqa: F403

plt, WONG = setup_plots()  # noqa: F405

reviews = load_reviews()      # noqa: F405
aspects = load_aspects()      # noqa: F405
features = load_features()    # noqa: F405
demand = load_demand()        # noqa: F405

# %% [markdown]
# ## 1. Volume — the primary, well-powered measure

# %%
women = aspects[aspects["aspect"] == "women_period"]
print(f"women_period mentions: {len(women):,}")
print(f"Distinct reviews:      {women['reviewId'].nunique():,} "
      f"({women['reviewId'].nunique()/len(reviews):.3%} of C_text)")
print(f"Apps represented:      {women['app_name'].nunique()} / {reviews['app_name'].nunique()}")

if women["reviewId"].nunique() < 200:
    print("\n>>> SCARCITY IS THE FINDING (§9.3). Report this n plainly in the paper.")

# %%
has_feature = set(features[(features["aspect"] == "women_period")
                           & (features["feature_present"] >= 1)]["app_name"])
print(f"Apps with Women tracking (N={len(has_feature)}): {sorted(str(a) for a in has_feature)}")

per_app = women.groupby("app_name").agg(
    mentions=("reviewId", "nunique"),
    pct_negative=("sentiment_label", lambda s: (s == "Negative").mean()),
).join(reviews.groupby("app_name").size().rename("n_reviews"))
per_app["mention_rate"] = per_app["mentions"] / per_app["n_reviews"]
per_app["has_feature"] = per_app.index.isin(has_feature)
display(per_app.sort_values("mentions", ascending=False).round(4))  # noqa: F821

# %% [markdown]
# ## 2. Sentiment on women-aspect mentions

# %%
display(sentiment_split(women["sentiment_label"]).round(3))  # noqa: F405,F821

women_app = women.assign(has_feature=women["app_name"].isin(has_feature))
display(women_app.groupby("has_feature").agg(  # noqa: F821
    mentions=("reviewId", "nunique"),
    apps=("app_name", "nunique"),
    pct_negative=("sentiment_label", lambda s: (s == "Negative").mean()),
    pct_positive=("sentiment_label", lambda s: (s == "Positive").mean()),
).round(3))

# %% [markdown]
# ## 3. Demand — do users of apps *without* the feature ask for it?

# %%
if len(demand):
    wd = demand[demand["aspect"] == "women_period"]
    without = wd[~wd["app_name"].isin(has_feature)]
    print(f"Demand signals for menstrual handling: {len(wd):,} total, "
          f"{len(without):,} in apps lacking the feature "
          f"({without['app_name'].nunique()} apps)")
    display(wd["request_type"].value_counts())  # noqa: F821
    print("\nSample requests:")
    for s in without["evidence_sentence"].head(10):
        print("  •", str(s)[:150])
else:
    print("demand.parquet not found.")

# %% [markdown]
# ## 4. Secondary: 3-vs-N comparison — UNDERPOWERED, descriptive only
#
# §9.3 names the confound directly: `Islamic Habit tracker` is one of the three
# and carries the lowest CSV rating in the corpus (3.8). Any apparent effect is
# as likely to be that app as the feature.

# %%
r = reviews.copy()
r["has_women_feature"] = r["app_name"].isin(has_feature)
comp = r.groupby("has_women_feature").agg(
    n_reviews=("reviewId", "size"), n_apps=("app_name", "nunique"),
    mean_star=("score", "mean"), mean_sentiment=("sent_num", "mean"),
)
display(comp.round(3))  # noqa: F821

a = r.loc[r["has_women_feature"], "score"].dropna().astype(float).to_numpy()
b = r.loc[~r["has_women_feature"], "score"].dropna().astype(float).to_numpy()
if len(a) > 5 and len(b) > 5:
    rng = np.random.default_rng(0)  # noqa: F405
    diffs = np.array([rng.choice(a, len(a), True).mean() - rng.choice(b, len(b), True).mean()  # noqa: F405
                      for _ in range(2000)])
    print(f"\nBootstrap Δ mean star = {diffs.mean():+.3f} "
          f"[{np.percentile(diffs, 2.5):+.3f}, {np.percentile(diffs, 97.5):+.3f}]")  # noqa: F405
    print(">>> N=3 treated apps. DESCRIPTIVE ONLY — no causal claim (§9.3).")

# %% [markdown]
# ## 5. Loyalty proxies
#
# Play data has no retention metric. Everything here is a **proxy** and every
# table caption must say so.

# %%
if len(demand):
    loyalty = (
        demand[demand["request_type"].isin(["tenure", "churn"])]
        .pivot_table(index="app_name", columns="request_type",
                     values="reviewId", aggfunc="nunique")
        .fillna(0)
        .join(reviews.groupby("app_name").size().rename("n_reviews"))
    )
    for c in ("tenure", "churn"):
        if c in loyalty:
            loyalty[f"{c}_rate"] = loyalty[c] / loyalty["n_reviews"]
    loyalty["thumbs_mean"] = reviews.groupby("app_name")["thumbsUpCount"].mean()
    loyalty["has_women_feature"] = loyalty.index.isin(has_feature)
    display(loyalty.sort_values("churn_rate", ascending=False).round(5))  # noqa: F821

# %% [markdown]
# ## 6. Privacy sentiment over time
#
# §10 pairs RQ4 with the privacy trajectory — bodily data and data-handling
# trust are the same concern for this user group.

# %%
priv = aspects[aspects["aspect"] == "privacy_data"]
pr = reviews[reviews["reviewId"].isin(set(priv["reviewId"]))].dropna(subset=["at"])
if len(pr):
    q = pr.set_index("at").groupby(pd.Grouper(freq="QE")).agg(  # noqa: F405
        n=("reviewId", "size"), mean_sentiment=("sent_num", "mean"))
    q = q[q["n"] >= 10]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(q.index, q["mean_sentiment"], color=WONG[4], lw=2, marker="o", ms=3)
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.axvline(pd.Timestamp("2020-11-16"), color=WONG[0], ls=":", lw=2)  # noqa: F405
    ax.annotate("Muslim Pro data-sale reporting", (pd.Timestamp("2020-11-16"), ax.get_ylim()[1]),  # noqa: F405
                xytext=(6, -14), textcoords="offset points", fontsize=8)
    ax.set_ylabel("Mean sentiment on privacy mentions")
    ax.set_title("Privacy sentiment over time (quarters with n ≥ 10)")
    fig.tight_layout()

# %%
if len(per_app):
    d = per_app.sort_values("mentions")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(d))))
    y = np.arange(len(d))  # noqa: F405
    ax.barh(y, d["mentions"], color=[WONG[2] if h else WONG[0] for h in d["has_feature"]])
    ax.set_yticks(y); ax.set_yticklabels([str(a)[:30] for a in d.index], fontsize=8)
    ax.set_xlabel("Reviews mentioning menstruation / exemptions")
    ax.set_title("RQ4 — women-aspect volume  (green = app has Women tracking)")
    fig.tight_layout()

# %% [markdown]
# ## 7. Qualitative coding set + quotes
#
# §9.3 asks for qualitative coding of the *full* set when it is small enough.

# %%
women_reviews = reviews[reviews["reviewId"].isin(set(women["reviewId"]))]
coding = women_reviews[["reviewId", "app_name", "score", "at", "content_clean"]].copy()
coding["theme_code"] = ""    # ACTION: fill during thematic analysis
coding.to_csv(DATA_DIR / "quotes" / "rq4_women_full_set.csv", index=False)  # noqa: F405
print(f"Wrote full women-aspect set for coding: {len(coding)} reviews")

export_quotes(women_reviews.nsmallest(15, "sent_num"), "rq4", n=15)  # noqa: F405

# %% [markdown]
# ## Answer to RQ4
#
# - women_period mention volume: ___ reviews across ___ apps
# - Sentiment split: ___% negative
# - Demand in apps lacking the feature: ___ signals
# - Between-app Δ star: ___ **(N=3, descriptive only)**
#
# → **Verdict**: _______
