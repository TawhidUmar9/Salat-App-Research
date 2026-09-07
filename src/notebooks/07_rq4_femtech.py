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
# Guarded: refuses to overwrite a sheet that already has theme_code filled in.
write_coding_sheet(coding, DATA_DIR / "quotes" / "rq4_women_full_set.csv")  # noqa: F405
print(f"Wrote full women-aspect set for coding: {len(coding)} reviews")

export_quotes(women_reviews.nsmallest(15, "sent_num"), "rq4", n=15)  # noqa: F405

# %% [markdown]
# ## Answer to RQ4
#
# **The scarcity is the finding.** Menstrual handling is mentioned in **2,182
# distinct reviews** (2,194 aspect mentions) across **24 of 26 apps** — 0.672% of
# the analysed corpus — yet only **three apps ship the feature**: Athan, Islamic
# Habit Tracker and Pillars. Users raise the subject across nearly the whole
# ecosystem; almost none of it accommodates them.
#
# **Demand is explicit and concentrated where the feature is absent.** Of 272
# demand signals, **216 come from the 16 apps that lack it**, and the request
# types are direct rather than incidental: 161 requests, 51 preference citations,
# 36 churn statements, 21 explicit-absence mentions.
#
# **Having the feature does not improve sentiment on the topic.** Mentions in the
# three apps that ship it run **26.3% negative** against **26.0%** in the 21 that
# do not — indistinguishable. The presence of menstrual handling is not
# associated with users being happier about menstrual handling, which suggests the
# implementations that exist are not meeting the need. The RQ1 coding supports
# this directly: `Menstrual mode unreliable` appears three times in the single-app
# pass and `Menstrual exemption absent` once in the cross-app pass, from different
# applications — first-hand evidence that the failure is in execution as well as
# in availability.
#
# **The between-app comparison is not estimable and is reported as such.** Apps
# with the feature average **4.135** stars against **4.371** without, Δ = −0.236.
# This must not be read as an effect. N = 3 on the treated side, and §9.3 names
# the confound outright: Islamic Habit Tracker is one of the three and carries the
# lowest rating in the corpus, so the difference is that application rather than
# the feature. M3 returns no standard error at all (`women_feature_score_diff`
# 0.085, flagged *descriptive, N=3 treated apps, underpowered*). Report it as not
# estimable — never as a null.
#
# **The topic model cannot add resolution here.** The `rq4_women_privacy`
# sub-model did not separate: after labelling, six of its ten topics carry the
# same theme — data selling to the US military — and a seventh is a near
# duplicate, putting 63% of the sub-model's documents under one label. The Muslim
# Pro privacy episode swamps this subset. Report themes, not the partition, and do
# not present ten topics as ten findings.
#
# → **Verdict**: Bio-spiritual inclusion is an unmet need rather than a contested
# design question. The demand is measurable, distributed across almost the entire
# ecosystem and voiced in the strongest available register — users leaving, users
# naming the absence — while the supply is three applications, none of which
# achieves better sentiment on the subject than the apps with no feature at all.
# Because only three apps have implemented it, no causal or comparative claim is
# possible and none is made here; the contribution is the size and shape of the
# gap, corroborated by qualitative evidence from two independently drawn coding
# passes. `women_period` is among the four aspects with under 25% ecosystem
# coverage and ranks second on the unmet-needs index (RQ6b), behind only companion
# hardware.
