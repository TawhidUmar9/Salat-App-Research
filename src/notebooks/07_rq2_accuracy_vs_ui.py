# ---
# jupyter:
#   jupytext:
#     text_representation: {extension: .py, format_name: percent}
# ---

# %% [markdown]
# # RQ2 — Algorithmic Accuracy vs. Pretty UI
#
# *Do calculation errors and bad compasses actually hurt ratings, or do users
# just want a nice interface?*
#
# Consumes M1 (§9.1). The answer is the **contrast** between the accuracy-family
# coefficients and `complaint_ui_design` — which class of complaint costs more
# stars, within the same app, controlling for review length and era.
#
# See `implementation_plan.md` §10, §9.1.

# %%
from _nbinit import *  # noqa: F403

plt, WONG = setup_plots()  # noqa: F405

reviews = load_reviews()        # noqa: F405
aspects = load_aspects()        # noqa: F405
results = load_model_results()  # noqa: F405

ACCURACY = ["prayer_times_accuracy", "qibla", "madhab", "calc_method"]
COMPARISON = ["ui_design", "stability_bugs", "ads_intrusive", "reminders_adhan"]

# %% [markdown]
# ## 1. Complaint volumes — how often is each raised at all?

# %%
neg = aspects[aspects["sentiment_label"] == "Negative"]
vol = (
    aspects.groupby("aspect")
    .agg(mentions=("reviewId", "nunique"),
         negative=("sentiment_label", lambda s: int((s == "Negative").sum())))
)
vol["neg_rate"] = vol["negative"] / vol["mentions"]
vol["family"] = np.where(vol.index.isin(ACCURACY), "accuracy",  # noqa: F405
                  np.where(vol.index.isin(COMPARISON), "comparison", "other"))  # noqa: F405
display(vol[vol["family"] != "other"].sort_values("mentions", ascending=False).round(3))  # noqa: F821

# %%
focus = vol[vol["family"] != "other"].sort_values("mentions")
fig, ax = plt.subplots(figsize=(9, 5))
colors = [WONG[4] if f == "accuracy" else WONG[0] for f in focus["family"]]
ax.barh(range(len(focus)), focus["mentions"], color=colors)
ax.set_yticks(range(len(focus))); ax.set_yticklabels(focus.index)
ax.set_xlabel("Reviews mentioning the aspect")
ax.set_title("Complaint volume: accuracy family (orange) vs. comparison (blue)")
for i, (n, r) in enumerate(zip(focus["mentions"], focus["neg_rate"])):
    ax.annotate(f"{r:.0%} neg", (n, i), xytext=(5, 0), textcoords="offset points",
                va="center", fontsize=8, color="grey")
fig.tight_layout()

# %% [markdown]
# ## 2. M1 coefficients — the answer
#
# Volume alone does not answer RQ2: a frequently-mentioned aspect may cost few
# stars. The mixed model is what separates "talked about" from "costly".

# %%
m1 = results[(results["model"] == "M1") & results["term"].str.startswith("complaint_", na=False)]
if m1.empty:
    print("No M1 results — run `python src/scripts/06_models.py --models m1`.")
else:
    m1 = m1.assign(aspect=m1["term"].str.replace("complaint_", "", regex=False))
    m1["family"] = np.where(m1["aspect"].isin(ACCURACY), "accuracy", "comparison")  # noqa: F405
    display(m1[["aspect", "family", "estimate", "ci_low", "ci_high",  # noqa: F821
                "p_value", "q_value"]].sort_values("estimate").round(4))

    acc_mean = m1.loc[m1["family"] == "accuracy", "estimate"].mean()
    ui = m1.loc[m1["aspect"] == "ui_design", "estimate"]
    print(f"\nAccuracy-family mean β: {acc_mean:+.4f} stars")
    if len(ui):
        print(f"UI complaint β:         {ui.iloc[0]:+.4f} stars")
        print(f"Ratio: accuracy complaints cost {acc_mean/ui.iloc[0]:.2f}× the UI penalty")

# %%
if not m1.empty:
    d = m1.sort_values("estimate")
    fig, ax = plt.subplots(figsize=(9, 5))
    y = range(len(d))
    ax.errorbar(d["estimate"], y,
                xerr=[d["estimate"] - d["ci_low"], d["ci_high"] - d["estimate"]],
                fmt="none", ecolor="grey", lw=1.2, capsize=3)
    ax.scatter(d["estimate"], y, s=60, zorder=3,
               c=[WONG[4] if f == "accuracy" else WONG[0] for f in d["family"]])
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(list(y)); ax.set_yticklabels(d["aspect"])
    ax.set_xlabel("Effect on star rating (95% CI)")
    ax.set_title("RQ2 — cost in stars per complaint type")
    fig.tight_layout()

# %% [markdown]
# ## 3. Qibla failure taxonomy
#
# §10 asks for a breakdown of *how* the compass fails, not just that it does.

# %%
qibla = aspects[aspects["aspect"] == "qibla"].copy()
txt = qibla["triggering_sentence"].fillna("").str.lower()
qibla["calibration"] = txt.str.contains(r"calibrat|figure\s*8|figure\s*eight|rotate", regex=True)
qibla["gps_location"] = txt.str.contains(r"gps|location|coordinate|map", regex=True)
qibla["wrong_direction"] = txt.str.contains(r"wrong|opposite|incorrect|off by|not accurate", regex=True)
qibla["spinning"] = txt.str.contains(r"spin|jump|unstable|keeps? moving|erratic", regex=True)
qibla["sensor"] = txt.str.contains(r"sensor|magnet|compass\s+not|no\s+compass", regex=True)

modes = ["calibration", "gps_location", "wrong_direction", "spinning", "sensor"]
tax = pd.DataFrame({  # noqa: F405
    "mentions": qibla[modes].sum(),
    "share_of_qibla": qibla[modes].mean(),
    "neg_rate": [qibla.loc[qibla[m], "sentiment_label"].eq("Negative").mean() for m in modes],
}).sort_values("mentions", ascending=False)
display(tax.round(3))  # noqa: F821

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(tax.index, tax["mentions"], color=WONG[4])
ax.set_ylabel("Mentions"); ax.set_title("Qibla failure modes")
plt.xticks(rotation=30, ha="right")
fig.tight_layout()

# %% [markdown]
# ## 4. The star/text divergence that motivates ABSA

# %%
high_star_ids = set(reviews.loc[reviews["score"] >= 4, "reviewId"])
acc_neg = aspects[
    aspects["aspect"].isin(ACCURACY)
    & (aspects["sentiment_label"] == "Negative")
    & aspects["reviewId"].isin(high_star_ids)
]
print(f"4–5★ reviews carrying a negative accuracy complaint: "
      f"{acc_neg['reviewId'].nunique():,}")
print("These are invisible to a star-only analysis — the §4.3 limitation, quantified.")
display(acc_neg["aspect"].value_counts())  # noqa: F821

# %% [markdown]
# ## 5. Quotes

# %%
acc_ids = set(aspects.loc[aspects["aspect"].isin(ACCURACY)
                          & (aspects["sentiment_label"] == "Negative"), "reviewId"])
ui_ids = set(aspects.loc[(aspects["aspect"] == "ui_design")
                         & (aspects["sentiment_label"] == "Negative"), "reviewId"])
sel = pd.concat([  # noqa: F405
    reviews[reviews["reviewId"].isin(acc_ids)].nsmallest(10, "sent_num"),
    reviews[reviews["reviewId"].isin(ui_ids)].nsmallest(5, "sent_num"),
])
export_quotes(sel, "rq2", n=15)  # noqa: F405

# %% [markdown]
# ## Answer to RQ2
#
# - Accuracy-family mean β = ___ stars; UI β = ___ stars
# - Dominant qibla failure mode: ___
# - High-star reviews hiding accuracy complaints: ___
#
# → **Verdict**: _______
