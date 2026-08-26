import pandas as pd, re, sys
from pathlib import Path
D = Path("src/data/gold_labels")
EXPECT = {"doc_500.csv":500,"doc_500_annotator1.csv":300,"doc_500_annotator2.csv":300,
          "aspect_300.csv":300,"demand_precision_sample.csv":200}
ACTION = ("gold_label","gold_aspect_correct","gold_aspect_true","gold_sentiment",
          "gold_is_request","is_true_positive","correct_aspect","annotator","notes")
ok = True
for f, n_exp in EXPECT.items():
    p = D/f
    if not p.exists():
        print(f"  MISSING {f}"); ok = False; continue
    d = pd.read_csv(p, dtype=str, keep_default_na=False)
    txt = next(c for c in ("content_clean","triggering_sentence","evidence_sentence") if c in d.columns)
    rows_ok = len(d) == n_exp
    src_ok  = d[txt].apply(lambda s: bool(re.search(r"[^\x00-\x7F]", str(s)))).sum()
    blank   = all((d[c].astype(str).str.strip()=="").all() for c in ACTION if c in d.columns)
    if "content_en" in d.columns:
        need = (d["lang"].ne("en") if "lang" in d.columns
                else d[txt].str.contains(r"[^\x00-\x7F]", regex=True, na=False))
        need &= d[txt].str.strip().ne("")
        got  = need & d["content_en"].str.strip().ne("")
        BAD = r"error\s*5\d\d|that.s an error|<html|server error|try again in \d+ seconds"
        poisoned = int(d["content_en"].str.contains(BAD, case=False, na=False).sum())
        tr = f"content_en {int(got.sum())}/{int(need.sum())}"
        # A poisoned value is a hard stop: an annotator would read it as the
        # review. A blank is not — ANNOTATION_GUIDE tells annotators to skip
        # untranslated rows and mark them, so a few are expected and fine.
        # Only a large shortfall means the translation pass genuinely failed.
        if poisoned:
            tr += f"  POISONED {poisoned}"; ok = False
        else:
            miss = int(need.sum()) - int(got.sum())
            cover = int(got.sum()) / max(1, int(need.sum()))
            if miss >= 5 and cover < 0.95:
                tr += f"  {miss} blank — RETRY"; ok = False
            elif miss:
                tr += f"  {miss} blank (ok, mark \'cannot read\')"
    else:
        tr = "no content_en"
    flag = "" if (rows_ok and blank) else "   <-- CHECK"
    if not (rows_ok and blank): ok = False
    print(f"  {f:31s} rows {len(d):>4}/{n_exp}  src-nonascii {src_ok:>3}  "
          f"labels {'blank' if blank else 'PRESENT'}  {tr}{flag}")
if not (D/"doc_500_annotator1.csv").exists() or not (D/"doc_500_annotator2.csv").exists():
    print("\n  Annotator files missing — run _gen_gold_sheets.py first.")
    sys.exit(1)
a1 = pd.read_csv(D/"doc_500_annotator1.csv", dtype=str, keep_default_na=False)
a2 = pd.read_csv(D/"doc_500_annotator2.csv", dtype=str, keep_default_na=False)
o1, o2 = set(a1[a1.is_overlap=="True"].reviewId), set(a2[a2.is_overlap=="True"].reviewId)
same = (len(o1)==100 and o1==o2)
leak = len(set(a1[a1.is_overlap!="True"].reviewId) & set(a2[a2.is_overlap!="True"].reviewId))
print(f"  overlap {len(o1)}/100 identical={o1==o2}  leakage={leak}")
if not same or leak: ok = False
print("\n  ALL CHECKS PASSED" if ok else "\n  SOMETHING IS OFF — do not start labelling yet")
sys.exit(0 if ok else 1)
