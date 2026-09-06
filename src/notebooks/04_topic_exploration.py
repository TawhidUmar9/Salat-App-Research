# ---
# jupyter:
#   jupytext:
#     text_representation: {extension: .py, format_name: percent}
# ---

# %% [markdown]
# # Phase 3 — Topic Exploration
#
# Interactive inspection and **manual labelling** of the BERTopic output.
# `04_topic_modeling.py` fits the model and writes `topic_info.csv` with an
# empty `manual_label` column; this notebook is where that column gets filled.
#
# See `implementation_plan.md` §7.

# %%
from _nbinit import *  # noqa: F403

plt, WONG = setup_plots()  # noqa: F405

reviews = load_reviews()   # noqa: F405
topics = pd.read_parquet(TOPICS_PATH)  # noqa: F405,F405
info = pd.read_csv(DATA_DIR / "topic_info.csv")  # noqa: F405

print(f"{len(info) - 1} topics over {len(topics):,} documents")
print(f"Unassigned (topic -1): {(topics['topic_id'] == -1).sum():,} "
      f"({(topics['topic_id'] == -1).mean():.1%})")

# %% [markdown]
# ## 1. Topic overview

# %%
display(info[["topic_id", "size", "top_words", "rq_tags", "manual_label"]].head(30))  # noqa: F821

# %%
d = info[info["topic_id"] >= 0].nlargest(25, "size").sort_values("size")
fig, ax = plt.subplots(figsize=(10, max(5, 0.34 * len(d))))
ax.barh(range(len(d)), d["size"], color=WONG[0])
ax.set_yticks(range(len(d)))
ax.set_yticklabels([f"{r.topic_id}: {str(r.top_words)[:44]}" for r in d.itertuples()], fontsize=7)
ax.set_xlabel("Documents")
ax.set_title("Topic sizes")
fig.tight_layout()

# %% [markdown]
# ## 2. Read representative documents
#
# Change `TOPIC` and re-run to work through the labelling pass.

# %%
TOPIC = 0

merged = topics.merge(reviews[["reviewId", "app_name", "score", "content_clean",
                               "roberta_label"]], on="reviewId", how="left")
sub = merged[merged["topic_id"] == TOPIC]
row = info[info["topic_id"] == TOPIC]
if len(row):
    print(f"Topic {TOPIC} — n={row.iloc[0]['size']:,}  RQ tags: {row.iloc[0]['rq_tags']}")
    print(f"Top words: {row.iloc[0]['top_words']}\n")
for i, r in enumerate(sub.head(15).itertuples(), 1):
    print(f"{i:2d}. [{r.score}★ {r.roberta_label}] {str(r.content_clean)[:200]}")

# %% [markdown]
# ## 3. Topic × sentiment
#
# Which topics are dominated by complaints?

# %%
assigned = merged[merged["topic_id"] >= 0]
ts = (assigned.groupby("topic_id")["roberta_label"]
      .value_counts(normalize=True).unstack(fill_value=0))
ts = ts.join(assigned.groupby("topic_id").size().rename("n"))
ts = ts.join(info.set_index("topic_id")[["top_words", "rq_tags"]])
display(ts.sort_values("Negative", ascending=False).head(20).round(3))  # noqa: F821

# %%
top = ts.nlargest(20, "n").sort_values("Negative")
fig, ax = plt.subplots(figsize=(10, max(5, 0.34 * len(top))))
left = np.zeros(len(top))  # noqa: F405
for lab, color in [("Negative", WONG[4]), ("Neutral", "#999999"), ("Positive", WONG[2])]:
    if lab in top.columns:
        ax.barh(range(len(top)), top[lab], left=left, color=color, label=lab)
        left += top[lab].to_numpy()
ax.set_yticks(range(len(top)))
ax.set_yticklabels([str(w)[:42] for w in top["top_words"]], fontsize=7)
ax.set_xlabel("Share"); ax.legend(ncol=3, loc="lower right")
ax.set_title("Sentiment composition of the 20 largest topics")
fig.tight_layout()

# %% [markdown]
# ## 4. Topic prevalence per app

# %%
prev = pd.read_csv(DATA_DIR / "topic_prevalence_by_app.csv")  # noqa: F405,F405
piv = prev.pivot_table(index="app_name", columns="topic_id", values="share_of_app", fill_value=0)
big = ts.nlargest(15, "n").index
piv = piv[[c for c in big if c in piv.columns]]

fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=90, fontsize=8)
ax.set_yticks(range(len(piv.index)))
ax.set_yticklabels([str(a)[:30] for a in piv.index], fontsize=8)
fig.colorbar(im, ax=ax, label="Share of that app's reviews", shrink=0.7)
ax.set_title("Topic prevalence per app"); ax.grid(False)
fig.tight_layout()

# %% [markdown]
# ## 5. Sub-topic models (RQ1 tracker, RQ4 women/privacy)

# %%
for name in ["rq1_tracker", "rq4_women_privacy", "rq5_bloat"]:
    p = DATA_DIR / f"topic_info_{name}.csv"  # noqa: F405
    if p.exists():
        print(f"\n=== {name} ===")
        display(pd.read_csv(p)[["topic_id", "size", "top_words",  # noqa: F405,F821
                                "representative_doc_1"]].head(12))
    else:
        print(f"{name}: not fitted (run 04_topic_modeling.py without --skip-subtopics)")

# %% [markdown]
# ## 6. Save manual labels
#
# **ACTION**: fill `LABELS` below from your reading, then run this cell. §7
# requires 15–25 interpretable themes named from representative documents.

# %%
LABELS = {
    # 0: "prayer time inaccuracy",
    # 1: "adhan notification failures",
    # 2: "streak motivation",
}

if LABELS:
    info["manual_label"] = info["topic_id"].map(LABELS).fillna(info["manual_label"])
    info.to_csv(DATA_DIR / "topic_info.csv", index=False)  # noqa: F405
    print(f"Saved {len(LABELS)} manual labels.")
else:
    print("LABELS is empty — fill it in and re-run this cell.")
