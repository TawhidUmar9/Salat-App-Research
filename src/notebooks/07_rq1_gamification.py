# ---
# jupyter:
#   jupytext:
#     text_representation: {extension: .py, format_name: percent}
# ---

# %% [markdown]
# # RQ1 — The Gamification Tension
#
# *Do users love or hate prayer streaks and grading scores?*
#
# Consumes M2 (§9.2). The claim this notebook must support or refute is that
# tracker sentiment is **bimodal** — a genuine love/hate split — rather than a
# lukewarm middle. A mean cannot distinguish those two, so the distribution
# shape and the guilt:motivation ratio carry the argument.
#
# See `implementation_plan.md` §10, §9.2.

# %%
from _nbinit import *  # noqa: F403

plt, WONG = setup_plots()  # noqa: F405

reviews = load_reviews()      # noqa: F405
aspects = load_aspects()      # noqa: F405
features = load_features()    # noqa: F405
results = load_model_results()  # noqa: F405

TRACKER_ASPECTS = ["prayer_tracker", "tracker_score", "goal_system"]

# %% [markdown]
# ## 1. Scope — how much tracker discourse is there?

# %%
tracker = aspects[aspects["aspect"].isin(TRACKER_ASPECTS)]
tracker_ids = set(tracker["reviewId"])
tr = reviews[reviews["reviewId"].isin(tracker_ids)].copy()

print(f"Tracker-aspect mentions: {len(tracker):,}")
print(f"Distinct reviews:        {len(tr):,}  ({len(tr)/len(reviews):.2%} of C_text)")
print(f"Apps represented:        {tr['app_name'].nunique()}")
print()
display(tracker.groupby("aspect").agg(  # noqa: F405,F821
    mentions=("reviewId", "nunique"),
    pct_negative=("sentiment_label", lambda s: (s == "Negative").mean()),
).round(3))

# %% [markdown]
# ## 2. M2 — sentiment by tracker tier

# %%
tier_map = {}
for app, grp in features.groupby("app_name"):
    has = dict(zip(grp["aspect"], grp["feature_present"].fillna(0)))
    tier_map[app] = ("tracker + score" if has.get("tracker_score", 0) >= 1
                     else "tracker only" if has.get("prayer_tracker", 0) >= 1 else "none")

tr["tracker_tier"] = tr["app_name"].map(tier_map).fillna("none")
tracker = tracker.assign(tier=tracker["app_name"].map(tier_map).fillna("none"))

summary = tr.groupby("tracker_tier").agg(
    n_reviews=("reviewId", "size"),
    n_apps=("app_name", "nunique"),
    mean_star=("score", "mean"),
    mean_sentiment=("sent_num", "mean"),
    pct_negative=("roberta_label", lambda s: (s == "Negative").mean()),
)
display(summary.round(3))  # noqa: F821

if len(results):
    display(results[results["model"] == "M2"][  # noqa: F821
        ["term", "estimate", "ci_low", "ci_high", "p_value", "q_value"]].round(4))

# %% [markdown]
# ## 3. Distribution shape — the actual RQ1 test
#
# §9.2: *"The love/hate tension is the ratio, not the mean sentiment — a mean
# can hide a bimodal split, which is precisely the phenomenon RQ1 posits."*

# %%
tracker_scored = tracker.dropna(subset=["prob_pos", "prob_neg"]).copy()
tracker_scored["valence"] = tracker_scored["prob_pos"] - tracker_scored["prob_neg"]

baseline = aspects[~aspects["aspect"].isin(TRACKER_ASPECTS)].dropna(subset=["prob_pos", "prob_neg"])
baseline_val = (baseline["prob_pos"] - baseline["prob_neg"]).to_numpy()
vals = tracker_scored["valence"].to_numpy()

print(f"Tracker valence: mean={vals.mean():+.3f}  sd={vals.std():.3f}  "
      f"%neg={np.mean(vals < -0.2):.1%}  %pos={np.mean(vals > 0.2):.1%}")  # noqa: F405
print(f"Baseline valence: mean={baseline_val.mean():+.3f}  sd={baseline_val.std():.3f}")

try:
    from diptest import diptest
    d_stat, d_p = diptest(vals)
    print(f"\nHartigan's dip test: D={d_stat:.4f}, p={d_p:.4g} → "
          f"{'BIMODAL' if d_p < 0.05 else 'not significantly bimodal'}")
except ImportError:
    from scipy.stats import kurtosis, skew
    n = len(vals)
    bc = (skew(vals) ** 2 + 1) / (kurtosis(vals) + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    print(f"\ndiptest not installed. Bimodality coefficient = {bc:.3f} (>0.555 → bimodal)")

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
axes[0].hist(vals, bins=50, color=WONG[0], alpha=0.8, density=True, label="Tracker aspects")
axes[0].hist(baseline_val, bins=50, color="grey", alpha=0.4, density=True, label="All other aspects")
axes[0].axvline(0, color="black", lw=1, ls="--")
axes[0].set_xlabel("Valence  P(pos) − P(neg)")
axes[0].set_ylabel("Density")
axes[0].set_title("Is tracker sentiment bimodal?")
axes[0].legend()

order = [t for t in ["none", "tracker only", "tracker + score"]
         if (tracker_scored["tier"] == t).sum() > 10]
data = [tracker_scored.loc[tracker_scored["tier"] == t, "valence"].to_numpy() for t in order]
parts = axes[1].violinplot(data, showmeans=True, showextrema=False)
for i, b in enumerate(parts["bodies"]):
    b.set_facecolor(WONG[i % len(WONG)]); b.set_alpha(0.7)
axes[1].set_xticks(range(1, len(order) + 1)); axes[1].set_xticklabels(order)
axes[1].axhline(0, color="black", lw=1, ls="--")
axes[1].set_ylabel("Valence")
axes[1].set_title("By tracker tier")
fig.tight_layout()

# %% [markdown]
# ## 4. Guilt vs. motivation
#
# The affective mechanism behind the split.

# %%
GUILT = r"\b(guilt|guilty|anxious|anxiety|pressure|ashamed|shame|stress|depress|fail|bad about)"
MOTIV = r"\b(motivat|encourag|discipline|consisten|barakah|closer to allah|helpful|improve|habit)"

affect = aspects[aspects["aspect"] == "spiritual_affect"].copy()
affect["in_tracker"] = affect["reviewId"].isin(tracker_ids)
txt = affect["triggering_sentence"].fillna("").str.lower()
affect["guilt"] = txt.str.contains(GUILT, regex=True)
affect["motivation"] = txt.str.contains(MOTIV, regex=True)

split = affect.groupby("in_tracker")[["guilt", "motivation"]].agg(["sum", "mean"])
display(split.round(4))  # noqa: F821

g_in = int(affect.loc[affect["in_tracker"], "guilt"].sum())
m_in = int(affect.loc[affect["in_tracker"], "motivation"].sum())
g_out = int(affect.loc[~affect["in_tracker"], "guilt"].sum())
m_out = int(affect.loc[~affect["in_tracker"], "motivation"].sum())
print(f"\nGuilt:motivation ratio  within tracker reviews: {g_in}:{m_in} = {g_in/max(1,m_in):.2f}")
print(f"Guilt:motivation ratio  elsewhere:               {g_out}:{m_out} = {g_out/max(1,m_out):.2f}")

if min(g_in, m_in, g_out, m_out) > 0:
    from scipy.stats import fisher_exact
    odds, p = fisher_exact([[g_in, m_in], [g_out, m_out]])
    print(f"Fisher exact OR = {odds:.2f}, p = {p:.4g}")

# %% [markdown]
# ## 5. Tracker sub-topics (from Phase 3)

# %%
if TOPICS_PATH.exists():  # noqa: F405
    topics = pd.read_parquet(TOPICS_PATH)  # noqa: F405
    sub_col = "rq1_tracker_topic"
    if sub_col in topics.columns:
        info_path = DATA_DIR / "topic_info_rq1_tracker.csv"  # noqa: F405
        if info_path.exists():
            display(pd.read_csv(info_path)[  # noqa: F405,F821
                ["topic_id", "size", "top_words", "manual_label"]].head(15))
    else:
        print("Sub-topic column not present — run 04_topic_modeling.py without --skip-subtopics.")
else:
    print("topics.parquet not found — run 04_topic_modeling.py.")

# %% [markdown]
# ## 6. Qualitative coding set + quotes

# %%
neg_tracker = (
    tr[tr["reviewId"].isin(set(tracker.loc[tracker["sentiment_label"] == "Negative", "reviewId"]))]
    .sort_values("sent_num")
)
print(f"Tracker-negative reviews available for coding: {len(neg_tracker):,}")

coding = neg_tracker.head(50)[["reviewId", "app_name", "score", "at", "content_clean"]].copy()
# Guarded: refuses to overwrite a sheet that already has theme_code filled in.
write_coding_sheet(coding, DATA_DIR / "quotes" / "rq1_tracker_negative_50.csv")  # noqa: F405

# Illustrative quotes: pull from both poles so the paper can show the split.
guilt_ids = set(affect.loc[affect["in_tracker"] & affect["guilt"], "reviewId"])
motiv_ids = set(affect.loc[affect["in_tracker"] & affect["motivation"], "reviewId"])
pole = pd.concat([  # noqa: F405
    tr[tr["reviewId"].isin(guilt_ids)].head(8),
    tr[tr["reviewId"].isin(motiv_ids)].head(7),
])
export_quotes(pole, "rq1", n=15)  # noqa: F405

# %% [markdown]
# ## Answer to RQ1
#
# Fill in once the numbers above are final:
#
# - Bimodality: dip test D=___, p=___
# - Guilt:motivation ratio inside tracker reviews vs. baseline: ___ vs ___
# - Tier effect (M2): β=___ [CI], q=___
#
# → **Verdict**: _______
