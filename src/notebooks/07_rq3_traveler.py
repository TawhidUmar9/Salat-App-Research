# ---
# jupyter:
#   jupytext:
#     text_representation: {extension: .py, format_name: percent}
# ---

# %% [markdown]
# # RQ3 — The Muslim Traveler Advantage
#
# *Does offering rare features like auto-shortening prayers give a competitive edge?*
#
# **Split question** (§2.1): Mosque Finder (8/20 apps) is testable as a
# between-app comparison. Auto Qasr (1/20) is not — with one treated app the
# causal question is unanswerable, so it is reframed as **demand evidence**:
# how many users of the apps *without* it ask for it, and how do users of the
# one app *with* it talk about it?
#
# See `implementation_plan.md` §10, §9.5.

# %%
from _nbinit import *  # noqa: F403

plt, WONG = setup_plots()  # noqa: F405

reviews = load_reviews()      # noqa: F405
aspects = load_aspects()      # noqa: F405
features = load_features()    # noqa: F405
demand = load_demand()        # noqa: F405
results = load_model_results()  # noqa: F405

# %% [markdown]
# ## 1. Who has what?

# %%
for aspect in ["mosque_finder", "qasr_travel"]:
    has = features[(features["aspect"] == aspect) & (features["feature_present"] >= 1)]
    total = features[features["aspect"] == aspect]["app_name"].nunique()
    print(f"{aspect:16s} present in {len(has):>2d}/{total} apps: "
          f"{', '.join(str(a)[:24] for a in has['app_name'])}")

# %% [markdown]
# ## 2. Preference citations — the "competitive edge" measure
#
# §6.2: patterns like *"that's why I use"*, *"only app that"*, *"no other app"*.

# %%
if len(demand):
    pref = demand[demand["request_type"] == "preference"]
    totals = reviews.groupby("app_name").size().rename("n_reviews")
    rate = (
        pref.groupby("app_name").size().rename("citations")
        .to_frame().join(totals, how="right").fillna({"citations": 0})
    )
    rate["citation_rate"] = rate["citations"] / rate["n_reviews"]
    display(rate.sort_values("citation_rate", ascending=False).head(15).round(5))  # noqa: F821

    print("\nWhich aspects do preference citations name?")
    display(pref["aspect"].value_counts().head(15))  # noqa: F821
else:
    print("demand.parquet not found — run 03c_demand_mining.py.")

# %% [markdown]
# ## 3. Mosque Finder — the testable half

# %%
has_mf = set(features[(features["aspect"] == "mosque_finder")
                      & (features["feature_present"] >= 1)]["app_name"])
mf = aspects[aspects["aspect"] == "mosque_finder"].copy()
mf["app_has_feature"] = mf["app_name"].isin(has_mf)

print(f"mosque_finder mentions: {len(mf):,} across {mf['app_name'].nunique()} apps")
display(mf.groupby("app_has_feature").agg(  # noqa: F821
    mentions=("reviewId", "nunique"),
    apps=("app_name", "nunique"),
    pct_negative=("sentiment_label", lambda s: (s == "Negative").mean()),
    pct_positive=("sentiment_label", lambda s: (s == "Positive").mean()),
).round(3))

if len(demand):
    mf_demand = demand[demand["aspect"] == "mosque_finder"]
    without = mf_demand[~mf_demand["app_name"].isin(has_mf)]
    print(f"\nMosque-finder demand in apps WITHOUT it: {len(without):,} signals "
          f"across {without['app_name'].nunique()} apps")

if len(results):
    display(results[results["model"] == "M5"].round(4))  # noqa: F821

# %% [markdown]
# ## 4. Auto Qasr — demand evidence, not a causal test
#
# §9.5: *"The **problem** is measurable ecosystem-wide even though the
# **solution** exists in one app."*

# %%
has_qasr = set(features[(features["aspect"] == "qasr_travel")
                        & (features["feature_present"] >= 1)]["app_name"])
qasr = aspects[aspects["aspect"] == "qasr_travel"]

print(f"qasr/travel mentions ecosystem-wide: {len(qasr):,} "
      f"across {qasr['app_name'].nunique()} apps")
print(f"Negative: {(qasr['sentiment_label'] == 'Negative').sum():,} "
      f"({(qasr['sentiment_label'] == 'Negative').mean():.1%})")

if len(demand):
    qd = demand[demand["aspect"] == "qasr_travel"]
    without = qd[~qd["app_name"].isin(has_qasr)]
    print(f"\nDemand signals in the {without['app_name'].nunique()} apps lacking it: {len(without):,}")
    display(without["request_type"].value_counts())  # noqa: F821

    per_app = without.groupby("app_name").agg(
        signals=("reviewId", "nunique")).join(
        reviews.groupby("app_name").size().rename("n_reviews"))
    per_app["rate"] = per_app["signals"] / per_app["n_reviews"]
    display(per_app.sort_values("rate", ascending=False).head(12).round(5))  # noqa: F821

# %% [markdown]
# ### The travel-failure narrative
#
# What actually goes wrong for travellers, in the apps that do not handle it.

# %%
qasr_neg = qasr[qasr["sentiment_label"] == "Negative"].copy()
txt = qasr_neg["triggering_sentence"].fillna("").str.lower()
qasr_neg["timezone_bug"] = txt.str.contains(r"time\s*zone|timezone|wrong\s+country|abroad", regex=True)
qasr_neg["manual_toggle"] = txt.str.contains(r"manual|every\s+time|have\s+to\s+change|reset", regex=True)
qasr_neg["no_qasr"] = txt.str.contains(r"no\s+qasr|doesn'?t\s+shorten|can'?t\s+shorten", regex=True)

modes = ["timezone_bug", "manual_toggle", "no_qasr"]
display(pd.DataFrame({  # noqa: F405,F821
    "mentions": qasr_neg[modes].sum(),
    "share": qasr_neg[modes].mean(),
}).round(3))

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

mf_rates = mf.groupby("app_name")["sentiment_label"].apply(lambda s: (s == "Negative").mean())
mf_counts = mf.groupby("app_name")["reviewId"].nunique()
keep = mf_counts[mf_counts >= 10].index
axes[0].scatter(mf_counts[keep], mf_rates[keep],
                c=[WONG[2] if a in has_mf else WONG[4] for a in keep], s=70)
axes[0].set_xscale("log")
axes[0].set_xlabel("Mosque-finder mentions (log)"); axes[0].set_ylabel("Negative rate")
axes[0].set_title("Mosque finder — green = app has the feature")

if len(demand):
    qd_app = (demand[demand["aspect"] == "qasr_travel"]
              .groupby("app_name")["reviewId"].nunique().sort_values().tail(12))
    axes[1].barh(range(len(qd_app)), qd_app.values,
                 color=[WONG[2] if a in has_qasr else WONG[4] for a in qd_app.index])
    axes[1].set_yticks(range(len(qd_app)))
    axes[1].set_yticklabels([str(a)[:24] for a in qd_app.index], fontsize=8)
    axes[1].set_xlabel("Qasr/travel demand signals")
    axes[1].set_title("Qasr demand — green = app already has it")
fig.tight_layout()

# %% [markdown]
# ## 5. Quotes

# %%
travel_ids = set(qasr["reviewId"]) | (set(demand.loc[demand["aspect"] == "qasr_travel", "reviewId"])
                                      if len(demand) else set())
sel = reviews[reviews["reviewId"].isin(travel_ids)].nsmallest(15, "sent_num")
export_quotes(sel, "rq3", n=15)  # noqa: F405

# %% [markdown]
# ## Answer to RQ3
#
# - Mosque finder: preference-citation rate with vs. without = ___ / ___
# - Qasr demand in the ___ apps lacking it: ___ signals
# - Dominant travel failure mode: ___
#
# → **Verdict**: _______ (state explicitly that the qasr half is demand
#   evidence, not a causal test — §2.1)
