"""
Recover hand labels from a gold-sheet backup after a pipeline re-run.

Re-running a stage regenerates its gold sheet as a NEW sample, so
write_gold_sheet() copies the filled version aside first. This merges those
labels back onto whatever rows the two files share.

Rows are matched on reviewId (plus request_type / predicted_aspect where a
review contributes several rows). Row ORDER is never assumed — the regenerated
sheet is a fresh draw and will not line up positionally.

Usage:
    python src/scripts/_merge_gold_backup.py \
        --backup src/data/gold_labels/demand_precision_sample.backup-<stamp>.csv \
        --into   src/data/gold_labels/demand_precision_sample.csv
    python src/scripts/_merge_gold_backup.py ... --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import log, section  # noqa: E402

#: Columns a human fills. Everything else comes from the regenerated sheet.
LABEL_COLS = ("is_true_positive", "correct_aspect", "notes", "gold_label",
              "gold_aspect_correct", "gold_aspect_true", "gold_sentiment",
              "gold_is_request", "annotator")

#: Extra keys that disambiguate several rows sharing one reviewId.
EXTRA_KEYS = ("request_type", "predicted_aspect", "evidence_sentence",
              "triggering_sentence")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backup", required=True, type=Path)
    p.add_argument("--into", required=True, type=Path)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    old = pd.read_csv(args.backup, dtype=str, keep_default_na=False)
    new = pd.read_csv(args.into, dtype=str, keep_default_na=False)
    section(f"Merging {args.backup.name} → {args.into.name}")

    keys = ["reviewId"] + [c for c in EXTRA_KEYS if c in old.columns and c in new.columns]
    labels = [c for c in LABEL_COLS if c in old.columns and c in new.columns]
    if not labels:
        raise SystemExit("No shared label columns between the two files.")
    log(f"matching on: {', '.join(keys)}")
    log(f"restoring:   {', '.join(labels)}")

    src = old[keys + labels].drop_duplicates(subset=keys)
    merged = new.merge(src, on=keys, how="left", suffixes=("", "_old"))

    restored = 0
    for c in labels:
        oldc = f"{c}_old"
        if oldc not in merged.columns:
            continue
        gap = merged[c].astype(str).str.strip().eq("") & merged[oldc].astype(str).str.strip().ne("")
        restored += int(gap.sum())
        merged.loc[gap, c] = merged.loc[gap, oldc]
        merged = merged.drop(columns=[oldc])

    key = next((c for c in ("is_true_positive", "gold_label", "gold_aspect_correct")
                if c in merged.columns), labels[0])
    have = int(merged[key].astype(str).str.strip().ne("").sum())
    log(f"{len(new):,} rows in the new sheet; {len(old):,} in the backup")
    log(f"recovered {restored:,} label cell(s); {have}/{len(merged)} rows now carry '{key}'")
    lost = len(merged) - have
    if lost:
        log(f"{lost} row(s) are new to this sample and still need labelling by hand.",
            level="WARN")

    if args.dry_run:
        log("--dry-run: nothing written.")
        return
    merged.to_csv(args.into, index=False)
    log(f"Wrote → {args.into}")


if __name__ == "__main__":
    main()
