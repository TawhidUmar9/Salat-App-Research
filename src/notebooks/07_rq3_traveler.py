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
# **Travel features are not what users cite when they explain their choice of
# app.** Across 4,696 preference citations — *"that's why I use"*, *"only app
# that"* — the aspects named are **adhan reminders (374), Quran audio (293),
# prayer-time accuracy (290), qibla (155)** and adhkar (152). Mosque finder does
# not appear in the top eight, and qasr does not appear at all. Preference-citation
# rates are near-identical whether an app has a mosque finder or not: **1.38%
# against 1.24%**, a ratio of 1.11 across 12 and 14 apps. Whatever earns loyalty in
# this ecosystem, it is not the travel feature set.
#
# **The mosque-finder coefficient runs the wrong way, and the reason matters.**
# The cluster-robust fit gives β = **−0.137** (p = 0.003, 21 app clusters,
# n = 3,010): apps that *have* a mosque finder attract **more negative** mentions
# of it — 27.0% negative against 23.0% in apps without it, with positives moving
# the other way (38.6% against 43.6%). The `mosque_finder` tagger runs at **27%**
# precision [10, 57], so roughly 820 of those 3,043 tagged mentions are genuine;
# the noise falls on both arms of the comparison, so the direction is more
# trustworthy than the volume, and neither should be reported without the
# precision figure. This should not be read as the feature
# harming apps. The two groups are not saying comparable things. In the 11 apps
# without a mosque finder, mentions are largely **requests** for one — 243 demand
# signals across 9 apps — and a request phrased as *"please add nearby mosques"*
# reads neutral-to-positive. In the 10 apps that have it, mentions are **usage
# reports**, and usage reports of a feature that fails read negative. You can only
# complain about something that exists. The 4-point gap measures the difference
# between wishing and using, not the cost of shipping.
#
# **The sign is unstable across specifications and is reported as such.** The
# mixed model returns +0.05 for the same term but with a NaN standard error, so it
# cannot arbitrate; the cluster-robust estimate is the trustworthy one. A
# coefficient that reverses direction between specifications, where one of them
# failed to produce an error estimate, does not support a directional claim about
# competitive advantage.
#
# **The qasr half is demand evidence and cannot be a causal test — N = 1.** Exactly
# **one app in twenty-six** implements travel concession (`Muslim Bangla Quran
# Hadith Dua`), so there is no between-app comparison to make. Demand is real but
# modest: **21 signals, 20 of them from 5 of the 25 apps lacking the feature** (10
# requests, 5 preference citations, 3 explicit absences, 3 churn statements). The
# requests are specific and practical — *"prayer times automatically adjust when
# I'm traveling because man forgets and my prayers be off"*, *"add namaz qasar for
# all 5 prayers … while anyone is out of city or in traveling"*, *"I wish it was
# installed prior my hajj trip"*.
#
# **Measurement caveat on the qasr aspect.** Its 636 mentions across 18 apps run
# **76.7% positive and only 7.1% negative**, which is implausible for a capability
# 25 of 26 apps lack. Only 3.3% of those mentions are demand signals. The lexicon
# is evidently catching travel and pilgrimage talk in general — hajj, umrah,
# journeys — rather than qasr functionality specifically. Treat the 636 as an
# upper bound on travel-related discourse, not as a measure of qasr discussion,
# and rest the qasr argument on the 21 demand signals, which were read directly.
#
# **Dominant travel failure mode:** prayer times not adjusting when the user
# changes location — the concession is not applied, and users discover it after
# the fact.
#
# → **Verdict**: Travel support does not confer a competitive edge. Users do not
# cite it when explaining why they chose an app, preference-citation rates are
# effectively equal with and without a mosque finder, and the one apparent
# association — more negative mentions in apps that have the feature — is an
# artefact of what the two groups are talking about rather than evidence of harm.
# The qasr half is explicitly **demand evidence, not a causal test**: with a single
# implementing app no between-app estimate is possible, and none is offered. What
# users actually cite as reasons for loyalty are adhan reminders, Quran audio and
# prayer-time accuracy — the core functions, two of which RQ6 identifies among the
# most frequently broken promises and one of which RQ2 shows to be the costliest
# accuracy failure. Competitive advantage in this ecosystem is earned by doing the
# basics reliably, not by adding travel features.
