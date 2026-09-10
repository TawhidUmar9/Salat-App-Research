# The Record, Not the Streak

Analysis code and manuscript for an ecosystem-scale study of user satisfaction
in Islamic prayer applications: 732,194 Google Play reviews across 26
applications, addressed through six pre-specified research questions.

## Layout

```
paper/        manuscript source — main.tex, sections/, figures/, references.bib
src/scripts/  the pipeline, run in numbered order (01_ … 08_)
src/notebooks/  per-research-question analysis, jupytext percent-format
src/data/     lexicon, gold sheets, topic sheets, derived outputs
reviews/      raw scrape, one directory per application
docs/         protocol, provenance and verification records
scrape_reviews.py   collection, driven by the feature-matrix CSV
```

## Building the paper

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Requires the ACM `acmart` class. The document class is
`[manuscript,review,anonymous]`, which is the CHI submission format and
suppresses the author block. Remove `anonymous` for camera-ready.

## Reproducing the analysis

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r src/requirements.txt
python src/scripts/01_preprocess.py       # then 02a, 02b, 03, 03b, 03c, 04, 05, 06, 08
```

The analysis reported in the paper was run on 2026-09-03 at commit `5da2db7`.
Re-running the modelling stage from a clean tree at that commit reproduces
`model_results.csv` byte-identically. All stochastic components are seeded; the
five seeds used for the topic-stability check (42–46) were fixed in advance.

`src/data/run_manifest.jsonl` records commit, hardware, batch size and library
versions for every reported figure. It is machine-specific and append-only, so
the copy that matters is the one from the machine the analysis ran on.

## Documentation

| File | What it is |
|---|---|
| `docs/implementation_plan.md` | the analysis plan, committed before the analysis was run |
| `docs/PAPER_DISCLOSURES.md` | what the paper must state, plus the pre-submission checklist |
| `docs/FACTS.md` | every reported number, with its status |
| `docs/ANNOTATION_GUIDE.md` | the hand-labelling protocol |
| `docs/MANUAL_WORK_GUIDE.md` | the manual passes and how they were drawn |
| `docs/EXECUTION_GUIDE.md` | running the pipeline end to end |
| `docs/review_fixups.txt` | diagnostic output behind the final round of corrections |

## Measurement caveats

The paper reports precision for every extraction stage rather than assuming the
tooling is correct. Seven of the 25 aspects with surviving gold rows fall below
50% precision, and counts for those are reported only as precision-adjusted
estimates with intervals. The guilt and motivation keyword lists used in RQ1
carry no gold sample at all, and the paper reports that comparison as a fact
about two lexicons rather than as a validated measurement.

## Data and reuse

The review text under `reviews/` belongs to the people who wrote it and is
subject to the Play Store's terms. It is included so the analysis can be
re-run, not for redistribution. If you publish anything derived from it, do not
reproduce reviewer identifiers, and quote sparingly.

Code and derived outputs are MIT licensed; see `LICENSE` for the scope.
