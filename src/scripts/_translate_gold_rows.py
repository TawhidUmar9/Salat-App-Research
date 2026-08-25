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
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import GOLD_DIR, log, section  # noqa: E402

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
    p.add_argument("--sleep", type=float, default=0.12,
                   help="Pause between calls, to stay under rate limits.")
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

        # Refuse to clobber translations already present and reviewed.
        if "content_en" in df.columns:
            already = df["content_en"].astype(str).str.strip().ne("").sum()
            if already:
                log(f"  {already} rows already carry content_en — leaving those alone.")
                mask &= df["content_en"].astype(str).str.strip().eq("")
                n = int(mask.sum())

        if args.dry_run or n == 0:
            continue

        if "content_en" not in df.columns:
            df["content_en"] = ""

        idx = df.index[mask]
        failed = 0
        for i, row_i in enumerate(idx, start=1):
            src = str(df.at[row_i, text_col])
            try:
                df.at[row_i, "content_en"] = translator.translate(src[:4500]) or ""
            except Exception as exc:  # one bad row must not lose the whole sheet
                failed += 1
                df.at[row_i, "content_en"] = ""
                if failed <= 3:
                    log(f"  row {row_i}: {type(exc).__name__}: {exc}", level="WARN")
            if i % 25 == 0:
                log(f"  {i}/{len(idx)}")
            time.sleep(args.sleep)

        df.to_csv(path, index=False)
        log(f"Wrote content_en for {len(idx) - failed}/{len(idx)} rows → {path}")
        if failed:
            log(f"{failed} row(s) failed and were left blank — re-run to retry.",
                level="WARN")

    if args.dry_run:
        section("Dry run")
        log(f"{grand_total:,} rows would be translated. No network calls were made.")
        log("Re-run without --dry-run to perform them.")


if __name__ == "__main__":
    main()
