"""
Phase 4 — Temporal Analysis
============================
Monthly/quarterly sentiment trends, changepoint detection,
and version-level analysis.

Input:
    - data/master_reviews_with_sentiment.parquet
    - data/aspect_sentiments.parquet

Output:
    - data/temporal_trends.parquet       (app × month aggregates)
    - data/changepoints.csv              (detected regime shifts)
    - data/aspect_volume_over_time.csv   (aspect × quarter)
    - data/version_sentiment.csv

See implementation_plan.md §8 for full specification.

Usage:
    python src/scripts/05_temporal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    ASPECT_PATH, DATA_DIR, SENTIMENT_PATH, TEMPORAL_PATH,
    base_parser, log, require, section, set_seed, summarize,
)

#: Muslim Pro data-sale reporting (Motherboard, Nov 2020) — the intervention
#: point for the §8 interrupted time-series.
MUSLIM_PRO_EVENT = pd.Timestamp("2020-11-16")

#: A month needs this many reviews before its mean is worth plotting (§8 confound).
MIN_MONTH_N = 20


def _sentiment_numeric(labels: pd.Series) -> pd.Series:
    """Map sentiment labels to -1/0/+1 so they can be averaged."""
    return labels.map({"Negative": -1.0, "Neutral": 0.0, "Positive": 1.0})


def compute_temporal_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    §8 — Monthly aggregates per app with a 3-month rolling mean.

    `n` is carried on every row deliberately: recency-weighted sampling (§2.4)
    makes early periods thin, and any trend plot must show n alongside the
    line so sparse-period noise is not read as signal.
    """
    section("§8  Temporal aggregates")

    d = df.dropna(subset=["at"]).copy()
    d["month"] = d["at"].dt.to_period("M").dt.to_timestamp()
    d["sent_num"] = _sentiment_numeric(d["roberta_label"])

    agg = (
        d.groupby(["app_name", "month"], observed=True)
        .agg(
            n=("reviewId", "size"),
            mean_score=("score", "mean"),
            mean_sentiment=("sent_num", "mean"),
            pct_negative=("roberta_label", lambda s: (s == "Negative").mean()),
            pct_positive=("roberta_label", lambda s: (s == "Positive").mean()),
            reply_rate=("has_reply", "mean"),
            median_words=("n_words", "median"),
        )
        .reset_index()
        .sort_values(["app_name", "month"])
    )

    for col in ("mean_score", "mean_sentiment", "pct_negative"):
        agg[f"{col}_roll3"] = (
            agg.groupby("app_name", observed=True)[col]
            .transform(lambda s: s.rolling(3, min_periods=2).mean())
        )

    agg["sparse"] = agg["n"] < MIN_MONTH_N

    log(f"{len(agg):,} app-month cells across {agg['app_name'].nunique()} apps")
    log(f"Coverage: {agg['month'].min():%Y-%m} → {agg['month'].max():%Y-%m}")
    log(f"Cells below n={MIN_MONTH_N} (flagged sparse): {agg['sparse'].sum():,} "
        f"({agg['sparse'].mean():.1%})")
    return agg


def detect_changepoints(series: pd.Series, *, penalty: float = 3.0, min_size: int = 6) -> list[int]:
    """
    §8 — PELT changepoint detection.

    Returns indices into `series` where the mean level shifts. Algorithmic
    detection rather than eyeballing: with 26 apps × 180 months, visual
    inspection is neither reproducible nor defensible in review.
    """
    vals = series.dropna().to_numpy()
    if len(vals) < 2 * min_size:
        return []
    try:
        import ruptures as rpt
    except ImportError:
        log("ruptures not installed — skipping changepoint detection.", level="WARN")
        return []

    algo = rpt.Pelt(model="rbf", min_size=min_size).fit(vals.reshape(-1, 1))
    bkps = algo.predict(pen=penalty)
    return [b for b in bkps if b < len(vals)]


def run_changepoint_scan(agg: pd.DataFrame, *, min_months: int = 24) -> pd.DataFrame:
    """Scan every app with enough history for sentiment and rating regime shifts."""
    section("§8  Changepoint scan")

    rows = []
    for app, grp in agg.groupby("app_name", observed=True):
        grp = grp[~grp["sparse"]].sort_values("month")
        if len(grp) < min_months:
            continue
        for metric in ("mean_sentiment", "mean_score"):
            idxs = detect_changepoints(grp[metric])
            for i in idxs:
                before = grp[metric].iloc[max(0, i - 6): i].mean()
                after = grp[metric].iloc[i: i + 6].mean()
                rows.append({
                    "app_name": app,
                    "metric": metric,
                    "changepoint_month": grp["month"].iloc[i],
                    "mean_before": before,
                    "mean_after": after,
                    "delta": after - before,
                    "n_after": int(grp["n"].iloc[i: i + 6].sum()),
                })

    out = pd.DataFrame(rows)
    if out.empty:
        log("No changepoints detected.")
        return out

    out = out.sort_values("delta")
    log(f"{len(out)} changepoints across {out['app_name'].nunique()} apps.")
    log("Largest sentiment drops:")
    drops = out[out["metric"] == "mean_sentiment"].head(10)
    for _, r in drops.iterrows():
        log(f"    {str(r['app_name'])[:32]:32s} {r['changepoint_month']:%Y-%m}  "
            f"Δ={r['delta']:+.3f}  (n={r['n_after']:,})")

    out.to_csv(DATA_DIR / "changepoints.csv", index=False)
    log(f"Saved → {DATA_DIR / 'changepoints.csv'}")
    return out


def muslim_pro_privacy_case_study(df: pd.DataFrame, aspect_df: pd.DataFrame) -> pd.DataFrame:
    """
    §8 — Muslim Pro 2020 privacy interrupted time-series.

    Outcome variables are privacy_data aspect *volume* (share of that month's
    reviews mentioning privacy) and its sentiment. Volume is the more robust
    of the two: the scandal should show up as people newly raising the topic,
    not only as existing mentions turning negative.
    """
    section("§8  Muslim Pro privacy case study")

    mp = df[df["app_name"].astype(str).str.contains("Muslim Pro", case=False, na=False)]
    if mp.empty:
        log("Muslim Pro not found in the corpus.", level="WARN")
        return pd.DataFrame()

    log(f"Muslim Pro reviews: {len(mp):,}")

    priv_ids = set(aspect_df.loc[aspect_df["aspect"] == "privacy_data", "reviewId"])
    d = mp.dropna(subset=["at"]).copy()
    d["month"] = d["at"].dt.to_period("M").dt.to_timestamp()
    d["is_privacy"] = d["reviewId"].isin(priv_ids)
    d["sent_num"] = _sentiment_numeric(d["roberta_label"])

    monthly = (
        d.groupby("month", observed=True)
        .agg(
            n=("reviewId", "size"),
            privacy_mentions=("is_privacy", "sum"),
            privacy_rate=("is_privacy", "mean"),
            mean_sentiment=("sent_num", "mean"),
            mean_score=("score", "mean"),
        )
        .reset_index()
    )
    monthly["period"] = np.where(monthly["month"] < MUSLIM_PRO_EVENT, "pre", "post")
    monthly["months_from_event"] = (
        (monthly["month"].dt.year - MUSLIM_PRO_EVENT.year) * 12
        + (monthly["month"].dt.month - MUSLIM_PRO_EVENT.month)
    )

    window = monthly[monthly["months_from_event"].between(-12, 12)]
    pre = window[window["period"] == "pre"]
    post = window[window["period"] == "post"]

    log(f"Event: {MUSLIM_PRO_EVENT:%Y-%m-%d} (Motherboard data-sale reporting)")
    log(f"±12-month window: {len(pre)} pre months ({pre['n'].sum():,} reviews), "
        f"{len(post)} post months ({post['n'].sum():,} reviews)")
    if len(pre) and len(post):
        log(f"  privacy mention rate  pre={pre['privacy_rate'].mean():.3%}  "
            f"post={post['privacy_rate'].mean():.3%}")
        log(f"  mean sentiment        pre={pre['mean_sentiment'].mean():+.3f}  "
            f"post={post['mean_sentiment'].mean():+.3f}")
        log(f"  mean star             pre={pre['mean_score'].mean():.2f}  "
            f"post={post['mean_score'].mean():.2f}")

        try:
            import statsmodels.formula.api as smf
            w = window.dropna(subset=["privacy_rate"]).copy()
            w["post"] = (w["period"] == "post").astype(int)
            w["time"] = w["months_from_event"]
            w["time_after"] = w["post"] * w["time"]
            # Standard ITS: level shift (post) + slope change (time_after).
            m = smf.wls("privacy_rate ~ time + post + time_after", data=w, weights=w["n"]).fit()
            log("")
            log("Interrupted time-series on privacy mention rate (weighted by n):")
            for term in ("post", "time_after"):
                log(f"    {term:12s} β={m.params[term]:+.5f}  p={m.pvalues[term]:.4f}")
            log("  NOTE: ITS assumes no other concurrent change. State this caveat.")
        except ImportError:
            log("statsmodels not installed — skipping the ITS fit.", level="WARN")

    monthly.to_csv(DATA_DIR / "muslim_pro_privacy_its.csv", index=False)
    log(f"Saved → {DATA_DIR / 'muslim_pro_privacy_its.csv'}")
    return monthly


def aspect_volume_over_time(df: pd.DataFrame, aspect_df: pd.DataFrame) -> pd.DataFrame:
    """
    §8 — Aspect mention volume per quarter.

    Directly informs RQ1: if gamification is a recent design trend, the
    tracker_score footprint should be dated, not flat across 2011–2026.
    Rates are shares of that quarter's reviews, not raw counts — raw counts
    only reproduce the sampling curve.
    """
    section("§8  Aspect volume over time")

    d = df.dropna(subset=["at"])[["reviewId", "at", "app_name"]].copy()
    d["quarter"] = d["at"].dt.to_period("Q").dt.to_timestamp()

    merged = aspect_df[["reviewId", "aspect", "sentiment_label"]].merge(d, on="reviewId", how="inner")

    denom = d.groupby("quarter", observed=True).size().rename("total_reviews")
    counts = (
        merged.groupby(["quarter", "aspect"], observed=True)
        .agg(
            mentions=("reviewId", "nunique"),
            pct_negative=("sentiment_label", lambda s: (s == "Negative").mean()),
        )
        .reset_index()
        .join(denom, on="quarter")
    )
    counts["mention_rate"] = counts["mentions"] / counts["total_reviews"]

    log("Aspect mention rate, first vs. last 3 years with data:")
    years = counts["quarter"].dt.year
    early = counts[years <= years.min() + 2].groupby("aspect", observed=True)["mention_rate"].mean()
    late = counts[years >= years.max() - 2].groupby("aspect", observed=True)["mention_rate"].mean()
    trend = pd.DataFrame({"early": early, "late": late}).dropna()
    trend["change"] = trend["late"] - trend["early"]
    for aspect, r in trend.sort_values("change", ascending=False).head(12).iterrows():
        log(f"    {str(aspect):24s} {r['early']:.2%} → {r['late']:.2%}  ({r['change']:+.2%})")

    counts.to_csv(DATA_DIR / "aspect_volume_over_time.csv", index=False)
    log(f"Saved → {DATA_DIR / 'aspect_volume_over_time.csv'}")
    return counts


def version_level_sentiment(df: pd.DataFrame, *, min_n: int = 30) -> pd.DataFrame:
    """
    §8 — Per-version sentiment, flagging releases with significant drops.

    appVersion coverage is spotty in Play scrapes; versions below `min_n`
    reviews are dropped rather than reported as noisy point estimates.
    """
    section("§8  Version-level sentiment")

    col = "reviewCreatedVersion" if "reviewCreatedVersion" in df.columns else "appVersion"
    if col not in df.columns:
        log("No version column present — skipping.", level="WARN")
        return pd.DataFrame()

    d = df[df[col].notna() & (df[col].astype(str).str.strip() != "")].copy()
    log(f"Reviews with a version tag: {len(d):,} / {len(df):,} ({len(d) / len(df):.1%})")
    if d.empty:
        return pd.DataFrame()

    d["sent_num"] = _sentiment_numeric(d["roberta_label"])
    out = (
        d.groupby(["app_name", col], observed=True)
        .agg(
            n=("reviewId", "size"),
            first_seen=("at", "min"),
            mean_score=("score", "mean"),
            mean_sentiment=("sent_num", "mean"),
            pct_negative=("roberta_label", lambda s: (s == "Negative").mean()),
        )
        .reset_index()
        .rename(columns={col: "version"})
    )
    out = out[out["n"] >= min_n].sort_values(["app_name", "first_seen"])

    out["app_median_sentiment"] = out.groupby("app_name", observed=True)["mean_sentiment"].transform("median")
    out["sentiment_delta"] = out["mean_sentiment"] - out["app_median_sentiment"]
    out["flagged_drop"] = out["sentiment_delta"] < -0.15

    log(f"{len(out)} versions with n>={min_n}; {int(out['flagged_drop'].sum())} flagged as drops.")
    for _, r in out[out["flagged_drop"]].sort_values("sentiment_delta").head(10).iterrows():
        log(f"    {str(r['app_name'])[:28]:28s} v{str(r['version'])[:12]:12s} "
            f"Δ={r['sentiment_delta']:+.3f}  n={r['n']:,}")

    out.to_csv(DATA_DIR / "version_sentiment.csv", index=False)
    log(f"Saved → {DATA_DIR / 'version_sentiment.csv'}")
    return out


def main() -> None:
    args = base_parser(__doc__ or "Phase 4 — temporal analysis").parse_args()
    set_seed(args.seed)

    require(SENTIMENT_PATH, "python src/scripts/02b_sentiment_roberta.py")
    df = pd.read_parquet(SENTIMENT_PATH)
    log(f"Loaded {len(df):,} reviews")

    if ASPECT_PATH.exists():
        aspect_df = pd.read_parquet(ASPECT_PATH)
        log(f"Loaded {len(aspect_df):,} aspect rows")
    else:
        log("aspect_sentiments.parquet not found — aspect-dependent sections will be skipped.", level="WARN")
        aspect_df = pd.DataFrame(columns=["reviewId", "aspect", "sentiment_label"])

    agg = compute_temporal_aggregates(df)
    run_changepoint_scan(agg)

    if len(aspect_df):
        muslim_pro_privacy_case_study(df, aspect_df)
        aspect_volume_over_time(df, aspect_df)

    version_level_sentiment(df)

    section("Writing output")
    agg.to_parquet(TEMPORAL_PATH, index=False, compression="zstd")
    summarize(agg, "temporal_trends")
    log(f"Saved → {TEMPORAL_PATH}")
    log("")
    log("Next: python src/scripts/06_models.py")


if __name__ == "__main__":
    main()
