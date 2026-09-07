# ---
# jupyter:
#   jupytext:
#     text_representation: {extension: .py, format_name: percent}
# ---

# %% [markdown]
# # RQ6 — Full-Fledged Spiritual Companion
#
# *Are current apps enough to meet all Salah-related needs?*
#
# The **synthesis** RQ. It consumes RQ1–RQ5 and splits into:
#
# ```
# RQ6a  DELIVERY GAP   feature claimed (✓) → do users complain?   "it exists but it's broken"
# RQ6b  COVERAGE GAP   feature absent  (✗) → do users request it? "it doesn't exist at all"
# ```
#
# The RQ6b ranking produced here is the **bridge to the Design Implications
# section** (§16.2) — stated in general ecosystem terms, not tied to any one app.

# %%
from _nbinit import *  # noqa: F403

plt, WONG = setup_plots()  # noqa: F405

reviews = load_reviews()      # noqa: F405
aspects = load_aspects()      # noqa: F405
features = load_features()    # noqa: F405
demand = load_demand()        # noqa: F405

gap = pd.read_csv(DATA_DIR / "gap_matrix.csv")           # noqa: F405
unmet = pd.read_csv(DATA_DIR / "unmet_needs_ranking.csv")  # noqa: F405
print(f"gap matrix: {len(gap):,} (app, feature) cells")

# %% [markdown]
# ## 1. RQ6a — Delivery gap
#
# Of the features an app claims, which do its users complain about?

# %%
promised = gap[gap["promised"].fillna(False)]
print(f"Promised (app, feature) pairs: {len(promised):,}")
print(f"With enough mentions to judge (n ≥ 30): {(promised['n_mentions'] >= 30).sum():,}")

broken = promised[promised["broken_promise"].fillna(False)].sort_values("neg_rate", ascending=False)
print(f"\nBROKEN PROMISES (>40% negative, n ≥ 30): {len(broken)}")
display(broken[["app_name", "aspect", "n_mentions", "n_negative", "neg_rate"]].round(3))  # noqa: F821

# %%
print("Which features are most often broken, across apps?")
by_feature = (
    promised[promised["n_mentions"] >= 30]
    .groupby("aspect")
    .agg(apps_promising=("app_name", "nunique"),
         apps_broken=("broken_promise", "sum"),
         mean_neg_rate=("neg_rate", "mean"),
         total_mentions=("n_mentions", "sum"))
)
by_feature["broken_share"] = by_feature["apps_broken"] / by_feature["apps_promising"]
display(by_feature.sort_values("broken_share", ascending=False).round(3))  # noqa: F821

# %%
piv = promised.pivot_table(index="app_name", columns="aspect", values="neg_rate")
n = promised.pivot_table(index="app_name", columns="aspect", values="n_mentions")
piv = piv.where(n >= 10)

fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(piv.values, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=90, fontsize=8)
ax.set_yticks(range(len(piv.index)))
ax.set_yticklabels([str(a)[:30] for a in piv.index], fontsize=8)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        if pd.notna(v) and v > 0.40:  # noqa: F405
            ax.text(j, i, "✕", ha="center", va="center", color="white", fontsize=7)
fig.colorbar(im, ax=ax, label="Negative rate on a promised feature", shrink=0.7)
ax.set_title("RQ6a — delivery gap (✕ = broken promise)")
ax.grid(False); fig.tight_layout()

# %% [markdown]
# ## 2. RQ6b — Coverage gap
#
# Where the feature is absent, how loudly do users ask for it?

# %%
display(unmet.sort_values("unmet_score", ascending=False).round(2))  # noqa: F821

# %%
d = unmet.sort_values("unmet_score").tail(15)
fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(d))))
y = np.arange(len(d))  # noqa: F405
ax.hlines(y, 0, d["unmet_score"], color="grey", lw=1.2)
ax.scatter(d["unmet_score"], y, s=80, color=WONG[4], zorder=3)
ax.set_yticks(y); ax.set_yticklabels(d["aspect"], fontsize=9)
ax.set_xlabel("Unmet-need score  (demand volume × apps lacking)")
ax.set_title("RQ6b — ranked unmet needs across the ecosystem")
fig.tight_layout()

# %% [markdown]
# ## 3. Synthesis across RQ1–RQ5
#
# One row per RQ with its headline number, so the discussion section can be
# written directly from this table.

# %%
def safe(fn, default="n/a"):
    try:
        return fn()
    except Exception:
        return default


tracker_ids = set(aspects.loc[aspects["aspect"].isin(
    ["prayer_tracker", "tracker_score", "goal_system"]), "reviewId"])
acc_ids = set(aspects.loc[aspects["aspect"].isin(
    ["prayer_times_accuracy", "qibla", "madhab", "calc_method"]), "reviewId"])

synthesis = pd.DataFrame([  # noqa: F405
    {"rq": "RQ1 gamification",
     "evidence": f"{len(tracker_ids):,} tracker reviews",
     "headline": safe(lambda: f"{aspects[aspects['aspect'].isin(['prayer_tracker','tracker_score']) ].pipe(lambda x: (x['sentiment_label']=='Negative').mean()):.1%} negative")},
    {"rq": "RQ2 accuracy vs UI",
     "evidence": f"{len(acc_ids):,} accuracy-aspect reviews",
     "headline": safe(lambda: f"{aspects[aspects['aspect'].isin(['prayer_times_accuracy','qibla'])].pipe(lambda x: (x['sentiment_label']=='Negative').mean()):.1%} negative")},
    {"rq": "RQ3 traveler",
     "evidence": safe(lambda: f"{len(demand[demand['aspect']=='qasr_travel']):,} qasr demand signals"),
     "headline": safe(lambda: f"{aspects[aspects['aspect']=='qasr_travel']['app_name'].nunique()} apps discussed")},
    {"rq": "RQ4 femtech",
     "evidence": safe(lambda: f"{aspects[aspects['aspect']=='women_period']['reviewId'].nunique():,} women-aspect reviews"),
     "headline": safe(lambda: f"{len(features[(features['aspect']=='women_period') & (features['feature_present']>=1)])} apps have the feature")},
    {"rq": "RQ5 bloat",
     "evidence": safe(lambda: f"{features.groupby('app_name')['feature_present'].sum().mean():.1f} mean features/app"),
     "headline": safe(lambda: f"{aspects[aspects['aspect']=='complexity_bloat']['reviewId'].nunique():,} bloat complaints")},
])
display(synthesis)  # noqa: F821

# %% [markdown]
# ## 4. The design-implications input
#
# **§16.2 decision**: the paper closes with a general Design Implications
# section derived from this ranking and stated in ecosystem terms. No specific
# app is named as the beneficiary.

# %%
top = unmet.sort_values("unmet_score", ascending=False).head(8)
print("Ranked unmet needs — the basis for the Design Implications section:\n")
for i, (_, r) in enumerate(top.iterrows(), 1):
    print(f"{i}. {r['aspect']}")
    print(f"     {r['demand_volume']:.0f} demand signals · {r['apps_lacking']:.0f} apps lack it "
          f"· {r['negative_mentions']:.0f} negative mentions where present")

    if len(demand):
        ex = demand[(demand["aspect"] == r["aspect"]) & (demand["request_type"] == "request")]
        for s in ex["evidence_sentence"].head(2):
            print(f'     "{str(s)[:120]}"')
    print()

# %% [markdown]
# ## 5. Combined gap view — is any need served *well* by anyone?

# %%
combined = (
    gap.groupby("aspect")
    .agg(apps_with=("promised", "sum"),
         apps_total=("app_name", "nunique"),
         mean_neg_rate=("neg_rate", "mean"),
         total_mentions=("n_mentions", "sum"),
         total_demand=("demand_signals", "sum"))
)
combined["coverage"] = combined["apps_with"] / combined["apps_total"]
combined["served_well"] = (combined["coverage"] > 0.5) & (combined["mean_neg_rate"] < 0.30)
display(combined.sort_values("coverage").round(3))  # noqa: F821

print(f"\nNeeds served well by most of the ecosystem: "
      f"{list(combined[combined['served_well']].index)}")
print(f"Needs served by NO app or served badly: "
      f"{list(combined[(combined['coverage'] < 0.25)].index)}")

# %% [markdown]
# ## 6. Quotes

# %%
if len(demand):
    top_aspects = set(top["aspect"])
    gap_ids = set(demand.loc[demand["aspect"].isin(top_aspects), "reviewId"])
    sel = reviews[reviews["reviewId"].isin(gap_ids)].head(15)
    export_quotes(sel, "rq6", n=15)  # noqa: F405

# %% [markdown]
# ## Answer to RQ6
#
# **RQ6a — the delivery gap.** Of 272 promised (app, feature) pairs, 136 carry
# enough mentions to judge (n ≥ 30), and **18 are broken promises across 11 apps**
# — a feature the app claims, drawing over 40% negative sentiment. Ranked by share
# of promising apps that break it: **monetization** (5 of 16 apps, 31%),
# **reminders/adhan** (5 of 20, 25%), mosque finder and forbidden times (1 of 4
# each), widgets (2 of 13), prayer-time accuracy (2 of 19). The largest by volume
# are Muslim Pro's reminders/adhan (9,989 mentions, 45.6% negative) and its
# monetization (9,182 mentions, 54.8%). `tracker_score` shows 1 of 1 broken, but
# with a single promising app that ratio carries no weight.
#
# **The two most-broken features are the two most basic.** Adhan reminders are the
# core function of a prayer-times app, and monetization is the terms on which it is
# offered. Failure concentrates not in advanced features but in the promise the
# app is built around.
#
# **RQ6b — the coverage gap.** Ranked by unmet-need score (demand volume ×
# apps lacking): **companion hardware** (6,725; 25 apps lacking), **menstrual
# handling** (4,738; 23 lacking), **mosque finder** (3,304; 14 lacking),
# **forbidden times** (2,805; 15 lacking), **nafl times** (2,445; 15 lacking).
# Four aspects fall below 25% ecosystem coverage: `calendar_sync`,
# `companion_hardware`, `qasr_travel` and `women_period`.
#
# **Some needs are served well.** `madhab`, `monetization`, `qibla`,
# `table_format` and `widgets` clear both bars — present in over half the
# ecosystem and averaging under 30% negative. Monetization appears on both lists,
# and the tension is informative rather than contradictory: most apps handle
# payment acceptably while a minority handle it badly enough to poison the
# feature, which is what a 31% break rate against a 25.8% mean negative rate
# describes.
#
# **Synthesis.** The gaps found here are the same ones the other questions arrive
# at independently. Companion hardware tops the unmet ranking while RQ5 finds
# hardware sentiment *better* inside the one app that ships a device (37.7%
# negative) than across the 23 that do not (49.2%), where mentions are unmet
# requests rather than complaints — demand, not dissatisfaction. Menstrual
# handling ranks second here and is RQ4's central finding. Reminders/adhan is the
# joint most-broken promise here and the dominant tracker failure theme in RQ1's
# cross-app coding, at 46%.
#
# → **Verdict — are current apps enough?** No, and they fall short in two
# different ways that call for different responses. The coverage gap is a market
# failure: entire categories of need — companion hardware, menstrual
# accommodation, travel concession, calendar integration — are served by under a
# quarter of the ecosystem despite measurable, explicitly voiced demand. The
# delivery gap is a quality failure, and a more damning one, because it falls on
# the promises apps have already made: the features most often broken are adhan
# reminders and monetization, the two things a prayer-times app most fundamentally
# offers. Building more features is not indicated. Delivering the promised ones
# reliably, and extending coverage to the populations currently unserved, is.
