"""
Add an English `content_en` column to the gold sheets so a human can read the
non-English rows.

Why this exists: doc_500 is stratified by language band, so it deliberately
oversamples non-English text — roughly half its rows are Arabic, Bengali,
Indonesian or Urdu, against 18% in the corpus as a whole. An annotator who
cannot read those rows must otherwise leave them blank, and 39 of the 100
inter-rater overlap rows are Arabic, so blanks there would gut Krippendorff's α.

01_preprocess.py's --translate-sample does the same job but samples randomly
across the corpus, before the gold sheets exist, so it will not reliably cover
these specific rows. This translates exactly the rows that need it.

The translations are a READING AID ONLY. They never enter the analysis: 02b
scores non-English text natively with twitter-xlm-roberta precisely because
machine translation flattens sentiment (implementation_plan.md §16.3). Tell
annotators to prefer a blank label over a guess when a translation reads
ambiguously.

NETWORK: this sends review text to Google Translate via deep-translator. The
text is already public on the Play Store, but the call leaves your machine, so
it is opt-in and never runs as part of the pipeline.

Usage:
    python src/scripts/_translate_gold_rows.py --dry-run     # count, no calls
    python src/scripts/_translate_gold_rows.py
    python src/scripts/_translate_gold_rows.py --only doc_500_annotator2.csv
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import GOLD_DIR, log, section  # noqa: E402

#: Google Translate answers a throttled request with an HTML error page, and
#: deep-translator returns that page BODY as if it were the translation — no
#: exception is raised. A first run at 0.12 s between calls wrote error pages
#: into 412 of 546 rows and reported complete success. Every response is now
#: checked against this before it is accepted.
BAD_RESPONSE = re.compile(
    r"error\s*5\d\d|that.s an error|<html|server error|"
    r"try again in \d+ seconds|request timed out|service unavailable",
    re.IGNORECASE,
)

#: Retry schedule for a rejected or failed call, in seconds.
BACKOFF = (2.0, 5.0, 12.0)


def latin_ratio(text: str) -> float:
    """Share of letters that are ASCII, ignoring digits, spaces and punctuation."""
    letters = re.sub(r"[\s\d\W_]+", "", str(text), flags=re.UNICODE)
    if not letters:
        return 1.0
    return sum(1 for ch in letters if ord(ch) < 128) / len(letters)


def looks_translated(src: str, out: str) -> bool:
    """Reject empty strings, error pages, and text that was never translated."""
    out = (out or "").strip()
    if not out or BAD_RESPONSE.search(out):
        return False
    # Google returns the input unchanged when it cannot detect or handle the
    # language — often for gibberish, or a script it gives up on. An exact-match
    # test is not enough: the reply can differ in whitespace yet still be the
    # original Arabic. Judge the OUTPUT instead — an English translation is
    # overwhelmingly Latin script, whatever the source was.
    if latin_ratio(out) < 0.5:
        return False
    return True


def translate_one(translator, text: str, sleep: float) -> str | None:
    """One row, with validation and backoff. None means give up for now."""
    for attempt, wait in enumerate((0.0, *BACKOFF)):
        if wait:
            time.sleep(wait)
        try:
            out = translator.translate(text[:4500])
        except Exception:
            continue
        if looks_translated(text, out):
            return out.strip()
    return None


#: sheet -> column holding the text a human must read
SHEETS = {
    "doc_500.csv": "content_clean",
    "doc_500_annotator1.csv": "content_clean",
    "doc_500_annotator2.csv": "content_clean",
    "aspect_300.csv": "triggering_sentence",
    "demand_precision_sample.csv": "evidence_sentence",
}


def needs_translation(df: pd.DataFrame, text_col: str) -> pd.Series:
    """Rows whose text is not English and is long enough to be worth reading."""
    if "lang" in df.columns:
        mask = df["lang"].notna() & (df["lang"].astype(str) != "en")
    else:
        # aspect_300 / demand sheets carry no lang column — fall back to script
        mask = df[text_col].astype(str).str.contains(r"[^\x00-\x7F]", regex=True, na=False)
    return mask & df[text_col].astype(str).str.strip().ne("")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true",
                   help="Report how many rows would be translated, make no calls.")
    p.add_argument("--only", metavar="FILE", help="Translate just one sheet.")
    p.add_argument("--sleep", type=float, default=1.0,
                   help="Pause between calls. Below ~0.5s Google throttles and returns error pages instead of translations.")
    args = p.parse_args()

    targets = {args.only: SHEETS[args.only]} if args.only else SHEETS

    translator = None
    if not args.dry_run:
        try:
            from deep_translator import GoogleTranslator
        except ImportError:
            raise SystemExit(
                "deep-translator is not installed.\n"
                "  uv pip install deep-translator\n"
                "Or re-run with --dry-run to see the row counts first."
            )
        translator = GoogleTranslator(source="auto", target="en")

    grand_total = 0
    for name, text_col in targets.items():
        path = GOLD_DIR / name
        if not path.exists():
            log(f"{name}: not found — skipping.", level="WARN")
            continue

        section(name)
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        if text_col not in df.columns:
            log(f"no '{text_col}' column — skipping.", level="WARN")
            continue

        mask = needs_translation(df, text_col)
        n = int(mask.sum())
        grand_total += n
        log(f"{len(df):,} rows; {n:,} need translation "
            f"({n / max(1, len(df)):.0%})")
        if "lang" in df.columns and n:
            top = df.loc[mask, "lang"].value_counts().head(6)
            log("  " + "  ".join(f"{k}={v}" for k, v in top.items()))

        # Keep good translations; re-do poisoned ones. An earlier run wrote
        # Google error pages into content_en, so "already present" is not the
        # same as "already done".
        if "content_en" in df.columns:
            cur = df["content_en"].astype(str)
            filled = cur.str.strip().ne("")
            poisoned = filled & (
                cur.str.contains(BAD_RESPONSE, na=False)
                | cur.apply(lambda s: latin_ratio(s) < 0.5)
            )
            good = filled & ~poisoned
            if int(poisoned.sum()):
                log(f"  {int(poisoned.sum())} rows hold an error page or "
                    f"untranslated text — clearing them for retry.", level="WARN")
                df.loc[poisoned, "content_en"] = ""
            if int(good.sum()):
                log(f"  {int(good.sum())} rows already translated — leaving those alone.")
            mask &= df["content_en"].astype(str).str.strip().eq("")
            n = int(mask.sum())
            log(f"  {n} rows to translate this pass")

        if args.dry_run or n == 0:
            continue

        if "content_en" not in df.columns:
            df["content_en"] = ""

        idx = df.index[mask]
        failed = 0
        for i, row_i in enumerate(idx, start=1):
            src = str(df.at[row_i, text_col])
            out = translate_one(translator, src, args.sleep)
            if out is None:
                failed += 1
                df.at[row_i, "content_en"] = ""     # blank, never a poisoned value
            else:
                df.at[row_i, "content_en"] = out
            if i % 25 == 0:
                log(f"  {i}/{len(idx)}  ({failed} failed so far)")
            # Save as we go: a throttled run that dies at row 400 should not
            # throw away the 399 translations it already earned.
            if i % 50 == 0:
                df.to_csv(path, index=False)
            time.sleep(args.sleep)

        df.to_csv(path, index=False)
        good = len(idx) - failed
        log(f"Wrote content_en for {good}/{len(idx)} rows → {path}")
        if failed:
            log(f"{failed} row(s) could not be translated and were left BLANK "
                f"(never a partial or error value). Re-run to retry just those; "
                f"raise --sleep if the failure rate is high.", level="WARN")

    if args.dry_run:
        section("Dry run")
        log(f"{grand_total:,} rows would be translated. No network calls were made.")
        log("Re-run without --dry-run to perform them.")


if __name__ == "__main__":
    main()
