"""
Shared infrastructure for the Salah-app analysis pipeline.
==========================================================
Paths, CLI conventions, device management, OOM-resilient batched inference,
and checkpoint/resume support.

Every script in scripts/ imports from here. Nothing in this module is
analysis-specific — it is plumbing.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

# HF's "Xet" chunked-storage backend reconstructs large files from many small
# content-addressed chunks; on this project's network that path pathologically
# stalls (dozens of small connections, the target blob never growing) while a
# plain sequential HTTP GET to the same CDN sustains 5-16 MB/s. Force the
# legacy download path before any HF-touching import runs. Must be set before
# `huggingface_hub` (imported transitively by `transformers`) is imported, so
# this sits at module import time, not inside a function.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import pandas as pd

# ─── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]      # src/
REPO_ROOT = PROJECT_ROOT.parent                          # falah-paper-codes/
REVIEWS_DIR = REPO_ROOT / "reviews"
CSV_PATH = REPO_ROOT / "Salah App Analysis - Sheet1.csv"

DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
GOLD_DIR = DATA_DIR / "gold_labels"
QUOTES_DIR = DATA_DIR / "quotes"
CKPT_DIR = DATA_DIR / "_checkpoints"
MODELS_DIR = DATA_DIR / "_models"

MASTER_PATH = DATA_DIR / "master_reviews.parquet"
SENTIMENT_PATH = DATA_DIR / "master_reviews_with_sentiment.parquet"
ASPECT_PATH = DATA_DIR / "aspect_sentiments.parquet"
LEXICON_PATH = DATA_DIR / "aspect_lexicon.yaml"
FEATURE_MATRIX_PATH = DATA_DIR / "feature_matrix_combined.parquet"
DEMAND_PATH = DATA_DIR / "demand.parquet"
TOPICS_PATH = DATA_DIR / "topics.parquet"
TEMPORAL_PATH = DATA_DIR / "temporal_trends.parquet"

for _d in (DATA_DIR, FIGURES_DIR, GOLD_DIR, QUOTES_DIR, CKPT_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ─── Logging ────────────────────────────────────────────────────────────────────

_T0 = time.time()


def log(msg: str, *, level: str = "INFO") -> None:
    """Timestamped stderr log so it interleaves correctly with tqdm bars."""
    elapsed = time.time() - _T0
    mm, ss = divmod(int(elapsed), 60)
    hh, mm = divmod(mm, 60)
    print(f"[{hh:02d}:{mm:02d}:{ss:02d}] {level:5s} {msg}", file=sys.stderr, flush=True)


def section(title: str) -> None:
    log("")
    log("─" * 70)
    log(title)
    log("─" * 70)


# ─── CLI ────────────────────────────────────────────────────────────────────────

def base_parser(description: str) -> argparse.ArgumentParser:
    """Argument parser preloaded with the flags every script shares."""
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--sample", type=int, default=None, metavar="N",
        help="Run on a random subsample of N reviews (dev mode). Omit for the full corpus.",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for sampling.")
    p.add_argument(
        "--batch-size", type=int, default=None,
        help="Inference batch size. Default: auto-sized from available VRAM.",
    )
    p.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu", "mps"],
        help="Compute device. 'auto' prefers CUDA, then MPS, then CPU.",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Reuse completed checkpoint shards instead of recomputing them.",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Delete this script's checkpoints before running.",
    )
    p.add_argument(
        "--fp32", action="store_true",
        help="Disable fp16/bf16 autocast (slower; use if you see NaNs).",
    )
    return p


def maybe_sample(df: pd.DataFrame, args: argparse.Namespace, *, by: str | None = "app_name") -> pd.DataFrame:
    """
    Apply --sample. Stratifies by `by` when present so a dev subsample still
    contains every app rather than only the largest one.
    """
    n = getattr(args, "sample", None)
    if not n or n >= len(df):
        return df
    if by and by in df.columns:
        out = stratified_sample(df, by, n, seed=args.seed)
        log(f"--sample {n}: stratified by {by} → {len(out):,} rows across {out[by].nunique()} apps")
        return out
    out = df.sample(n, random_state=args.seed).reset_index(drop=True)
    log(f"--sample {n}: random → {len(out):,} rows")
    return out


def stratified_sample(df: pd.DataFrame, by: str, n: int, *, seed: int = 42) -> pd.DataFrame:
    """
    Take roughly `n` rows spread evenly across the levels of `by`.

    Written without `groupby.apply` on purpose: the `include_groups` argument
    changed meaning in pandas 2.2 and was removed in 3.0, and this needs to run
    on both. Small groups contribute everything they have; the shortfall is
    topped up at random so the result lands close to `n`.
    """
    if by not in df.columns or n >= len(df):
        return df.reset_index(drop=True)

    groups = [g for _, g in df.groupby(by, observed=True)]
    per = max(1, n // max(1, len(groups)))
    parts = [g.sample(min(len(g), per), random_state=seed) for g in groups]
    out = pd.concat(parts, ignore_index=True) if parts else df.head(0)

    if len(out) < n:
        remainder = df.drop(index=pd.concat(parts).index, errors="ignore")
        take = min(n - len(out), len(remainder))
        if take > 0:
            out = pd.concat([out, remainder.sample(take, random_state=seed)], ignore_index=True)
    elif len(out) > n:
        out = out.sample(n, random_state=seed)
    return out.reset_index(drop=True)


# ─── Device ─────────────────────────────────────────────────────────────────────

def get_device(preference: str = "auto") -> tuple[str, float]:
    """
    Resolve the compute device and its usable memory in GB.

    Returns (device_str, memory_gb). memory_gb is 0.0 for CPU, which callers
    treat as "use the conservative batch size".
    """
    try:
        import torch
    except ImportError:
        log("torch not installed — falling back to CPU-only code paths.", level="WARN")
        return "cpu", 0.0

    if preference in ("cuda", "auto") and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gb = props.total_memory / 1024**3
        log(f"Device: cuda — {props.name} ({gb:.1f} GB VRAM)")
        record_run_fact("device", "cuda")
        record_run_fact("gpu", props.name)
        record_run_fact("vram_gb", round(gb, 1))
        return "cuda", gb
    if preference in ("mps", "auto") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        log("Device: mps (Apple Silicon)")
        return "mps", 0.0
    if preference == "cuda":
        log("--device cuda requested but CUDA is unavailable; using CPU.", level="WARN")
    log("Device: cpu — transformer stages will be slow. Consider --sample for a dry run.")
    return "cpu", 0.0


#: Rough per-model VRAM cost in GB for one sequence of ~128 tokens in fp16,
#: used to auto-size batches. Deliberately conservative.
_MODEL_FOOTPRINT = {
    "base": 0.006,    # roberta-base / deberta-v3-base class (~125M params)
    "large": 0.022,   # bart-large-mnli class (~400M params)
    "embed": 0.004,   # MiniLM sentence encoders
}


def auto_batch_size(memory_gb: float, model_class: str = "base", *, requested: int | None = None) -> int:
    """
    Pick a batch size that fits in VRAM, leaving ~35% headroom for weights,
    activations and fragmentation. Explicit --batch-size always wins.
    """
    if requested:
        return requested
    if memory_gb <= 0:
        return 16  # CPU / MPS
    weights = {"base": 0.6, "large": 1.8, "embed": 0.3}[model_class]
    usable = max(0.5, memory_gb * 0.65 - weights)
    bs = int(usable / _MODEL_FOOTPRINT[model_class])
    bs = max(4, min(512, 1 << (bs.bit_length() - 1)))  # clamp + round down to power of 2
    log(f"Auto batch size for '{model_class}' on {memory_gb:.1f} GB → {bs}")
    # Auto-sizing means the batch depends on whatever GPU is present, and batch
    # shape can perturb bf16 reductions enough to flip a borderline zero-shot
    # decision. Record it so Methods can state the value actually used.
    record_run_fact(f"batch_size_{model_class}", bs)
    return bs


def autocast_dtype(device: str, force_fp32: bool = False):
    """Best available reduced precision for this device."""
    import torch
    if force_fp32 or device != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


# ─── OOM-resilient batching ─────────────────────────────────────────────────────

def _is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error" in msg or isinstance(exc, MemoryError)


def batched(seq: Sequence, size: int) -> Iterator[Sequence]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def run_with_oom_backoff(
    fn: Callable[[Sequence], object],
    items: Sequence,
    batch_size: int,
    *,
    desc: str = "inference",
    min_batch: int = 1,
) -> list:
    """
    Map `fn` over `items` in batches, halving the batch size and retrying
    whenever CUDA reports OOM. Returns the concatenated list of results.

    `fn` receives a list of items and must return a list of the same length.
    """
    from tqdm.auto import tqdm

    results: list = []
    bs = batch_size
    idx = 0
    pbar = tqdm(total=len(items), desc=desc, unit="item", file=sys.stderr)
    while idx < len(items):
        chunk = items[idx: idx + bs]
        try:
            out = fn(chunk)
        except Exception as exc:  # noqa: BLE001 — we re-raise anything that is not OOM
            if not _is_oom(exc) or bs <= min_batch:
                pbar.close()
                raise
            _empty_cache()
            bs = max(min_batch, bs // 2)
            log(f"CUDA OOM — halving batch size to {bs} and retrying.", level="WARN")
            continue
        if len(out) != len(chunk):
            pbar.close()
            raise RuntimeError(f"{desc}: fn returned {len(out)} results for {len(chunk)} inputs")
        results.extend(out)
        idx += len(chunk)
        pbar.update(len(chunk))
    pbar.close()
    return results


def _empty_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def sort_by_length(texts: Sequence[str]) -> tuple[list[str], np.ndarray]:
    """
    Length-bucket texts so each batch pads to a similar length.

    Returns (sorted_texts, inverse_index). Apply `inverse_index` to the
    results to restore the original order:  results[inverse] .
    """
    order = np.argsort([len(t) for t in texts], kind="stable")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return [texts[i] for i in order], inverse


# ─── Checkpointing ──────────────────────────────────────────────────────────────

class Checkpoint:
    """
    Shard-based checkpointing for long GPU runs.

    Each shard is a parquet file under data/_checkpoints/<name>/. A run with
    --resume skips shards that already exist; a crash therefore costs at most
    one shard of work.

    `config` fingerprints the settings that determine *what* gets computed. It
    is stored beside the shards and re-checked on resume: if it has changed, the
    shards are stale and are cleared automatically rather than being mixed with
    freshly computed ones.

    That invalidation is not cosmetic. Shards are keyed by *position* in the
    deduplicated work list, so changing something like --zeroshot-threshold
    changes which items fall in which shard. Resuming across that boundary makes
    cached shard N describe different rows than new shard N, and items missing
    from the cache silently merge to NaN instead of being scored — wrong numbers,
    no error.
    """

    def __init__(self, name: str, *, resume: bool = False, overwrite: bool = False,
                 config: dict | None = None):
        self.dir = CKPT_DIR / name
        self.name = name
        self.config = config
        cfg_path = self.dir / "_config.json"

        if overwrite and self.dir.exists():
            for f in self.dir.glob("*.parquet"):
                f.unlink()
            log(f"Checkpoint '{name}': cleared.")

        if resume and config is not None and cfg_path.exists():
            try:
                prior = json.loads(cfg_path.read_text())
            except Exception:
                prior = None
            if prior != config:
                changed = sorted(
                    k for k in set(config) | set(prior or {})
                    if (prior or {}).get(k) != config.get(k)
                )
                log(f"Checkpoint '{name}': settings changed since the cached run "
                    f"({', '.join(changed)}) — discarding stale shards.", level="WARN")
                for k in changed:
                    log(f"    {k}: {(prior or {}).get(k)!r} -> {config.get(k)!r}", level="WARN")
                for f in self.dir.glob("*.parquet"):
                    f.unlink()

        self.dir.mkdir(parents=True, exist_ok=True)
        self.resume = resume
        if config is not None:
            cfg_path.write_text(json.dumps(config, sort_keys=True, default=str))

    def has(self, shard: str | int) -> bool:
        return self.resume and (self.dir / f"{shard}.parquet").exists()

    def write(self, shard: str | int, df: pd.DataFrame) -> None:
        df.to_parquet(self.dir / f"{shard}.parquet", index=False)

    def read(self, shard: str | int) -> pd.DataFrame:
        return pd.read_parquet(self.dir / f"{shard}.parquet")

    def collect(self) -> pd.DataFrame:
        files = sorted(self.dir.glob("*.parquet"))
        if not files:
            return pd.DataFrame()
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    def clear(self) -> None:
        for f in self.dir.glob("*.parquet"):
            f.unlink()


# ─── App-name / package helpers ─────────────────────────────────────────────────

def normalize_app_key(name: str) -> str:
    """
    Compact, lossless-ish key for matching app names across sources.

    Deliberately NOT aggressive: an earlier version stripped domain words
    ("prayer", "times", "qibla"...) which collapsed *Prayer Alarm* and
    *Prayer Time, Azan Alarm, Qibla* onto the same key and silently mis-joined
    them. Here we only case-fold and drop non-alphanumerics, so distinct names
    stay distinct. Fuzzy resolution is handled by `resolve_app_packages`, which
    uses the package id as the authoritative key.
    """
    if not isinstance(name, str):
        return ""
    return re.sub(r"[^\w]", "", name.lower(), flags=re.UNICODE)


def resolve_app_packages(
    csv_names: pd.Series, csv_packages: pd.Series, meta: pd.DataFrame,
) -> pd.Series:
    """
    Resolve each annotation-sheet row to a canonical package id.

    The package id (`appId` in metadata.json) is the only identifier stable
    across sources — store titles drift, and the sheet records a few apps by a
    short name that no longer matches the listing. Resolution order:

      1. the package the sheet already records,
      2. exact normalized-title match,
      3. substring match in either direction on the normalized title,
      4. unresolved (NaN) — reported, never guessed.

    Returns a Series of package ids aligned to `csv_names`.
    """
    def lead(name: str) -> str:
        """Normalized brand segment, before any ':' or ' - ' subtitle."""
        return normalize_app_key(re.split(r"[:\-–—,]", str(name), maxsplit=1)[0])

    valid = set(meta["package"].dropna())
    by_key = {normalize_app_key(t): p for t, p in zip(meta["app_name"], meta["package"])}
    by_lead: dict[str, list] = {}
    for t, p in zip(meta["app_name"], meta["package"]):
        by_lead.setdefault(lead(t), []).append(p)

    out = []
    for name, pkg in zip(csv_names, csv_packages):
        if isinstance(pkg, str) and pkg.strip() in valid:
            out.append(pkg.strip())
            continue

        key = normalize_app_key(name)
        if key in by_key:
            out.append(by_key[key])
            continue

        hits = [p for k, p in by_key.items() if key and (key in k or k in key)]
        if len(hits) == 1:
            out.append(hits[0])
            continue

        # Subtitles drift independently on the store ("Sajda: Quran, Athan,
        # Prayer" vs "Sajda: Prayer Times, Quran"), so fall back to the brand
        # segment — but only when it identifies exactly one app.
        lk = lead(name)
        lead_hits = [p for k, ps in by_lead.items() if lk and (lk in k or k in lk) for p in ps]
        out.append(lead_hits[0] if len(set(lead_hits)) == 1 else None)

    resolved = pd.Series(out, index=csv_names.index, dtype="object")
    n_bad = int(resolved.isna().sum())
    if n_bad:
        log(f"{n_bad} annotation rows could not be matched to a package: "
            f"{list(csv_names[resolved.isna()])}", level="WARN")
    return resolved


def load_metadata() -> pd.DataFrame:
    """
    Load metadata.json for every app directory under reviews/.

    Returns one row per app with the fields the plan uses as app-level
    covariates, plus the raw description for Phase 2.5 promise extraction.
    """
    rows = []
    for d in sorted(REVIEWS_DIR.iterdir()):
        meta_file = d / "metadata.json"
        if not d.is_dir() or not meta_file.exists():
            continue
        with open(meta_file, encoding="utf-8") as f:
            m = json.load(f)
        rows.append({
            "app_dir": d.name,
            "app_name": m.get("title") or d.name,
            "package": m.get("appId"),
            "description": m.get("description") or "",
            "summary": m.get("summary") or "",
            "realInstalls": m.get("realInstalls"),
            "app_score": m.get("score"),
            "app_ratings": m.get("ratings"),
            "app_reviews_total": m.get("reviews"),
            "released": m.get("released"),
            "lastUpdatedOn": m.get("lastUpdatedOn"),
            "adSupported": bool(m.get("adSupported")),
            "offersIAP": bool(m.get("offersIAP")),
            "free": bool(m.get("free", True)),
            "genre": m.get("genre"),
            "developer": m.get("developer"),
        })
    df = pd.DataFrame(rows)
    df["released_dt"] = pd.to_datetime(df["released"], format="mixed", errors="coerce")
    # Package id is the canonical join key everywhere downstream.
    df["app_key"] = df["package"]
    return df


# ─── Feature-matrix (CSV) parsing ───────────────────────────────────────────────

#: Ordered feature columns as they appear in the annotation sheet.
FEATURE_COLUMNS = [
    "Prayer Times by Location",
    "Madhab Variations",
    "Various methods of calculation",
    "Timely Reminders",
    "Forbidden Times",
    "Nafl Prayer Times",
    "Waqt relative reminders",
    "Custom Reminder Sounds",
    "Goal System and Other events (streak)",
    "Mosque Finder",
    "Salah and/or Wudu Guides",
    "Salah Specific Adhkars",
    "Qibla Compass",
    "Prayer times in Table format",
    "Auto Qasr mode",
    "Connect to Google Calendar",
    "All Features for free",
    "Prayer Tracker",
    "Tracker Analysis / Score",
    "Women tracking",
    "Widgets",
    "Has Companion Hardware",
    # Added when the sheet was completed to 26/26. They carry no aspect mapping
    # (no review-side lexicon entry corresponds to them), so they contribute to
    # `feature_count` for RQ5 but not to the promise/delivery matching in 03b.
    "Life Qaza Calculator",
    "Habit builder and tracker",
]

#: aspect name -> CSV feature column, per implementation_plan.md §5.1.
ASPECT_TO_FEATURE = {
    "prayer_times_accuracy": "Prayer Times by Location",
    "madhab": "Madhab Variations",
    "calc_method": "Various methods of calculation",
    "reminders_adhan": "Timely Reminders",
    "forbidden_times": "Forbidden Times",
    "nafl_times": "Nafl Prayer Times",
    "goal_system": "Goal System and Other events (streak)",
    "mosque_finder": "Mosque Finder",
    "guides": "Salah and/or Wudu Guides",
    "adhkar": "Salah Specific Adhkars",
    "qibla": "Qibla Compass",
    "table_format": "Prayer times in Table format",
    "qasr_travel": "Auto Qasr mode",
    "calendar_sync": "Connect to Google Calendar",
    "monetization": "All Features for free",
    "prayer_tracker": "Prayer Tracker",
    "tracker_score": "Tracker Analysis / Score",
    "women_period": "Women tracking",
    "widgets": "Widgets",
    "companion_hardware": "Has Companion Hardware",
}
FEATURE_TO_ASPECT = {v: k for k, v in ASPECT_TO_FEATURE.items()}

#: 20 CSV-linked aspects (§5.1) + 7 review-only aspects (§5.2).
CSV_LINKED_ASPECTS = list(ASPECT_TO_FEATURE.keys())
REVIEW_ONLY_ASPECTS = [
    "ui_design", "stability_bugs", "ads_intrusive", "privacy_data",
    "quran_audio", "complexity_bloat", "spiritual_affect",
]
ALL_ASPECTS = CSV_LINKED_ASPECTS + REVIEW_ONLY_ASPECTS

#: Natural-language glosses used as NLI hypotheses in the zero-shot backstop
#: (§5.3 step 3) and in promise extraction (§6.1). Phrasing matters: these are
#: substituted into "This review is about {}." so they must read as noun phrases.
ASPECT_DESCRIPTIONS = {
    "prayer_times_accuracy": "the accuracy of the prayer times",
    "madhab": "the madhab or school of jurisprudence used for prayer timing",
    "calc_method": "the prayer time calculation method or angle settings",
    "reminders_adhan": "the adhan notification, alarm or prayer reminder",
    "forbidden_times": "the forbidden or makruh times for prayer",
    "nafl_times": "optional nafl prayers such as tahajjud or duha",
    "goal_system": "goals, challenges, badges or rewards",
    "mosque_finder": "finding a nearby mosque or congregation",
    "guides": "a guide teaching how to pray or perform wudu",
    "adhkar": "dhikr, dua or tasbih content",
    "qibla": "the qibla compass and its direction",
    "table_format": "a monthly timetable or calendar view of prayer times",
    "qasr_travel": "shortening prayers while travelling, and timezone handling",
    "calendar_sync": "syncing prayer times to a calendar app",
    "monetization": "the price, subscription, paywall or in-app purchases",
    "prayer_tracker": "logging or tracking which prayers were performed",
    "tracker_score": "prayer streaks, scores, statistics or progress charts",
    "women_period": "menstruation and prayer exemptions for women",
    "widgets": "home screen or lock screen widgets",
    "companion_hardware": "a companion smartwatch, ring or wearable device",
    "ui_design": "the user interface design and visual appearance",
    "stability_bugs": "crashes, bugs, freezing or performance problems",
    "ads_intrusive": "advertisements and how intrusive they are",
    "privacy_data": "privacy, data collection and permissions",
    "quran_audio": "Quran text, recitation or translation",
    "complexity_bloat": "the app having too many features or being cluttered",
    "spiritual_affect": "the emotional or spiritual effect of using the app",
}


#: Cell values that mean "annotator checked, feature is absent" — as opposed to
#: a blank (absence by omission) or any other text (a note, hence presence).
ABSENT_MARKERS = frozenset({"x", "✗", "✘", "-", "–", "—", "n", "no", "0", "false"})


def load_feature_csv() -> pd.DataFrame:
    """
    Parse the hand-annotation sheet.

    The sheet has three spacer rows above the header, and encodes presence as
    a tick ('✓') *or* a free-text note (e.g. 'Google Maps', 'Tahajjud').

    Absence is encoded EITHER as an explicit marker ('x') OR as a blank cell.
    Both are read as "verified absent"; a row is treated as annotated if it
    holds any non-empty cell at all. Distinguishing these matters: an explicit
    'x' means the annotator checked and found nothing, whereas a blank is only
    absence-by-omission. Treating 'x' as presence — as this function did before
    the sheet was completed with explicit markers — silently marks every feature
    present for every app and collapses `feature_count` to a constant, which
    destroys RQ5's independent variable without raising an error.

    Apps with no annotation at all are returned with NaN across every feature —
    never 0 — so downstream joins can distinguish "verified absent" from
    "not yet annotated".
    """
    raw = pd.read_csv(CSV_PATH, header=3, encoding="utf-8-sig", dtype=str)
    raw = raw.dropna(how="all")
    raw = raw[raw["App Name"].notna() & (raw["App Name"].str.strip() != "")]

    present = [c for c in FEATURE_COLUMNS if c in raw.columns]
    missing = set(FEATURE_COLUMNS) - set(present)
    if missing:
        log(f"Feature columns absent from CSV: {sorted(missing)}", level="WARN")

    out = pd.DataFrame({
        "app_name_csv": raw["App Name"].str.strip(),
        "package": raw["package name"].str.strip() if "package name" in raw.columns else pd.NA,
        "csv_rating": pd.to_numeric(raw.get("Rating"), errors="coerce"),
        "reviewer": raw.get("Reviewer"),
        "summary_note": raw.get("General Review Summary"),
    })

    vals = raw[present].apply(lambda s: s.fillna("").str.strip())
    filled = vals != ""
    absent = vals.apply(lambda s: s.str.lower().isin(ABSENT_MARKERS))
    binary = filled & ~absent
    n_filled = binary.sum(axis=1)
    annotated = filled.any(axis=1)

    n_absent = int(absent.sum().sum())
    log(f"Feature CSV encoding: {int(binary.sum().sum()):,} present, "
        f"{n_absent:,} explicit-absent ('x'), "
        f"{int((~filled).sum().sum()):,} blank.")
    if n_absent == 0:
        log("No explicit-absent markers found — absence is encoded as blanks only.",
            level="WARN")

    for col in present:
        out[col] = np.where(annotated, binary[col].astype(float), np.nan)
    for col in missing:
        out[col] = np.nan

    out["is_annotated"] = annotated.values
    out["feature_count"] = np.where(annotated, n_filled, np.nan)

    # Resolve to the canonical package id so every downstream join is on a
    # stable key rather than on drifting store titles.
    meta = load_metadata()
    out["package"] = resolve_app_packages(out["app_name_csv"], out["package"], meta)
    out = out.merge(
        meta[["package", "app_name"]].rename(columns={"app_name": "app_name_meta"}),
        on="package", how="left",
    )
    out["app_key"] = out["package"]

    log(f"Feature CSV: {len(out)} apps — {int(annotated.sum())} annotated, "
        f"{int((~annotated).sum())} unannotated (features left as NaN).")
    log(f"Resolved to packages: {out['package'].notna().sum()}/{len(out)}")
    return out.reset_index(drop=True)


# ─── Lexicon ────────────────────────────────────────────────────────────────────

def load_lexicon() -> dict[str, dict]:
    """Load and validate data/aspect_lexicon.yaml."""
    import yaml
    with open(LEXICON_PATH, encoding="utf-8") as f:
        lex = yaml.safe_load(f)
    for aspect, spec in lex.items():
        spec.setdefault("keywords", [])
        spec.setdefault("regex_patterns", [])
        spec.setdefault("csv_column", None)
        spec.setdefault("rqs", [])
    log(f"Lexicon: {len(lex)} aspects, "
        f"{sum(len(v['keywords']) for v in lex.values())} keywords, "
        f"{sum(len(v['regex_patterns']) for v in lex.values())} regex patterns.")
    return lex


def compile_aspect_patterns(
    lexicon: dict[str, dict], *, variant: str = "review",
) -> dict[str, re.Pattern]:
    """
    Build one compiled alternation regex per aspect.

    Keywords are matched on word boundaries so 'ads' does not fire inside
    'adhan'; multi-word keywords allow flexible internal whitespace. Explicit
    regex_patterns from the YAML are appended verbatim.

    `variant` selects the vocabulary:

      "review"  — the default `keywords` / `regex_patterns`, tuned for how
                  users talk in reviews ("prayers are wrong when I travel").

      "promise" — `promise_keywords` / `promise_regex_patterns` where an aspect
                  defines them, falling back to the review set otherwise. Store
                  descriptions are marketing copy and need a stricter, more
                  distinctive vocabulary: the review-side keyword `journey`
                  matched "embark on a spiritual journey" in 13 apps' listings
                  and inflated `qasr_travel` from 1 promising app to 14, which
                  would have collapsed RQ3's unmet-need score and fabricated
                  RQ6a broken promises.
    """
    kw_key = "promise_keywords" if variant == "promise" else "keywords"
    rx_key = "promise_regex_patterns" if variant == "promise" else "regex_patterns"

    compiled, overridden = {}, []
    for aspect, spec in lexicon.items():
        keywords = spec.get(kw_key) or spec["keywords"]
        regexes = spec.get(rx_key) or (spec["regex_patterns"] if kw_key not in spec or not spec.get(kw_key) else [])
        if variant == "promise" and spec.get(kw_key):
            overridden.append(aspect)

        parts = []
        for kw in keywords:
            esc = re.escape(kw.strip().lower()).replace(r"\ ", r"\s+")
            parts.append(rf"\b{esc}\b" if kw.strip() else "")
        parts.extend(regexes)
        parts = [p for p in parts if p]
        if not parts:
            log(f"Aspect '{aspect}' has no patterns — it will never match.", level="WARN")
            continue
        compiled[aspect] = re.compile("|".join(parts), re.IGNORECASE | re.UNICODE)

    if overridden:
        log(f"Promise variant: {len(overridden)} aspects use a stricter vocabulary "
            f"({', '.join(sorted(overridden))})")
    return compiled


# ─── Misc ───────────────────────────────────────────────────────────────────────

# ─── Reproducibility manifest ───────────────────────────────────────────────────
#
# Every run appends one JSON line to data/run_manifest.jsonl recording the exact
# command, the resolved git commit, the device and batch size actually used, and
# the versions of every library that can move a number. CHI artifact review asks
# how a figure was produced; "we ran the script" is not an answer when the batch
# size is auto-derived from whatever GPU happened to be present.
#
# Registered via atexit so no per-script wiring is needed and a crashed run still
# leaves a record of what it was attempting.

_RUN_FACTS: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%S")}
_RUN_T0 = time.time()


def record_run_fact(key: str, value) -> None:
    """Note something the manifest should remember about this run."""
    _RUN_FACTS[key] = value


def _library_versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for mod in ("torch", "transformers", "pandas", "numpy", "statsmodels",
                "sklearn", "bertopic", "hdbscan", "umap", "pysbd", "ruptures"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # not installed, or exposes no __version__
            pass
    return out


def _git_state() -> dict:
    import subprocess
    def _run(*a):
        try:
            return subprocess.run(a, cwd=PROJECT_ROOT, capture_output=True, text=True,
                                  timeout=5).stdout.strip()
        except Exception:
            return ""
    sha = _run("git", "rev-parse", "HEAD")
    dirty = bool(_run("git", "status", "--porcelain"))
    return {"commit": sha or "unknown", "dirty_working_tree": dirty}


def _write_run_manifest() -> None:
    try:
        entry = {
            **_RUN_FACTS,
            "script": Path(sys.argv[0]).name,
            "command": " ".join(sys.argv),
            "duration_sec": round(time.time() - _RUN_T0, 1),
            "git": _git_state(),
            "versions": _library_versions(),
        }
        with (DATA_DIR / "run_manifest.jsonl").open("a") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
    except Exception:  # a manifest must never be the reason a run fails
        pass


atexit.register(_write_run_manifest)


def require(path: Path, produced_by: str) -> Path:
    """Fail fast with an actionable message when an upstream stage has not run."""
    if not path.exists():
        raise SystemExit(
            f"\nMissing input: {path}\n"
            f"Run `{produced_by}` first.\n"
        )
    return path


def write_gold_sheet(sheet: pd.DataFrame, path: Path, action_cols: Sequence[str]) -> Path:
    """
    Write a hand-labelling sheet, never silently destroying existing labels.

    Every sheet here is regenerated by re-running its producing stage, and the
    sheets represent days of human annotation. If `path` already exists and any
    ACTION column holds a filled value, the old file is copied aside to
    `<stem>.backup-<timestamp>.csv` before the new one is written, and the
    backup is logged loudly. Unfilled sheets are overwritten without ceremony.
    """
    if path.exists():
        try:
            prior = pd.read_csv(path)
            filled = sum(
                int(prior[c].notna().sum() - (prior[c].astype(str).str.strip() == "").sum())
                for c in action_cols if c in prior.columns
            )
        except Exception:  # unreadable/corrupt — back it up rather than judge it
            filled = -1
        if filled != 0:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            backup = path.with_name(f"{path.stem}.backup-{stamp}.csv")
            backup.write_bytes(path.read_bytes())
            log(f"EXISTING LABELS PRESERVED → {backup}", level="WARN")
            log(f"  {path.name} held {filled if filled >= 0 else 'unreadable'} filled "
                f"cell(s) across {', '.join(action_cols)}. The regenerated sheet is a "
                f"NEW sample — merge on reviewId, do not assume row order matches.",
                level="WARN")

    sheet.to_csv(path, index=False)
    return path


def summarize(df: pd.DataFrame, name: str) -> None:
    log(f"{name}: {len(df):,} rows × {len(df.columns)} cols "
        f"({df.memory_usage(deep=True).sum() / 1024**2:.0f} MB)")


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_coding_sheet(df: "pd.DataFrame", path: Path, code_col: str = "theme_code") -> Path:
    """
    Write a qualitative coding sheet without destroying anyone's coding.

    These sheets are regenerated every time their producing script or notebook
    runs, but the whole point of them is the column a human fills in by hand.
    Overwriting a coded sheet silently discards hours of work that cannot be
    reproduced, so refuse instead and say where the coded file is.
    """
    import pandas as pd  # local: _common is imported before pandas elsewhere

    if path.exists():
        try:
            existing = pd.read_csv(path)
        except Exception:  # noqa: BLE001 — unreadable file is not worth protecting
            existing = None
        if existing is not None and code_col in existing.columns:
            filled = existing[code_col].fillna("").astype(str).str.strip()
            n = int((filled != "").sum())
            if n:
                log(f"{path.name} already has {n} coded rows — NOT overwriting. "
                    f"Delete it first if you really want a fresh sheet.", level="WARN")
                return path

    out = df.copy()
    if code_col not in out.columns:
        out[code_col] = ""
    out.to_csv(path, index=False)
    log(f"Wrote coding sheet ({len(out)} rows) → {path}")
    return path
