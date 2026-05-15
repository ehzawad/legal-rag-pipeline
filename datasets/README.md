# Dataset

20 public-domain (or synthetic) legal-style documents: 15 organized by
the five "messy document" characteristics from the take-home spec, plus
a 5-doc clean true-PDF control set for plain end-to-end checks:

> "Expect scanned pages, low-resolution PDFs, handwritten notes,
> partially illegible records, and inconsistently formatted files."

Each messy category has **3 representative documents**, all sliced to
≤6 pages and ≤1.3 MB so a single run touches every code path without
burning budget on multi-hundred-page PDFs. The `truepdf/` control set
has 5 official U.S. Courts forms, each ≤5 pages.

## Layout

```
datasets/
├── scanned-pages/                    3 docs  — scanned/fax-style sources, mixed text layers
├── low-resolution-pdfs/              3 docs  — sub-200 DPI sources
├── handwritten-notes/                3 docs  — cursive primary content
├── partially-illegible-records/      3 docs  — visible text-layer mojibake
├── inconsistently-formatted-files/   3 docs  — heterogeneous layouts
└── truepdf/                          5 docs  — clean text-layer court forms
```

## Run against any category

The pipeline walks the input dir recursively. For day-to-day checks, point
at one source-document folder such as `datasets/truepdf` or
`datasets/handwritten-notes`.

```bash
uv run pipeline run \
  --input  datasets/handwritten-notes \
  --output outputs/handwritten_notes \
  --case-id handwritten-notes \
  --task   "Summarize the obligations and parties named in these documents."

uv run pipeline run \
  --input  datasets \
  --output outputs/all_categories \
  --case-id all-categories \
  --task   "Summarize key obligations across the supplied documents."

uv run pipeline run \
  --input  datasets/truepdf \
  --output outputs/truepdf_clean \
  --case-id truepdf-clean \
  --task   "Review these clean U.S. Courts form PDFs and prepare a plain-English document summary. Identify what each form is for, any explicit deadlines or actions stated on the forms, and any fields that are blank or require a human to fill in. Do not invent parties, case names, dates, or obligations that are not filled in."
```

See [`COVERAGE.md`](COVERAGE.md) for per-document inspection and the
cross-category mapping (most documents exercise more than one
characteristic — the placement here reflects each doc's *primary*
contribution).

## Sources

All documents are public-domain, CC-licensed, or synthetic:
- Internet Archive (pre-1923 US legal manuscripts)
- SEC EDGAR VPRR series (US federal government, paper-filing scans)
- UCSF Industry Documents Library (court-released litigation exhibits)
- Library of Congress Chronicling America (microfilm newspapers)
- CUAD (CC BY 4.0)
- US Supreme Court slip opinion (US federal government)
- Synthetic noisy lien notice (built deterministically by `synthetic/_build_noisy_doc.py`,
  not shipped here — the rendered PDF is in `low-resolution-pdfs/`)
- U.S. Courts official forms in `truepdf/`

Evaluation pipeline inputs live in top-level [`../eval`](../eval), not in
this dataset folder. That includes the harness manifest, gold labels,
public download manifest, eval-point checks, and example operator edits.
