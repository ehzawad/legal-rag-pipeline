# Dataset coverage matrix

How every shipped document maps to the five "messy document" categories
from the take-home spec.

> "Expect scanned pages, low-resolution PDFs, handwritten notes,
> partially illegible records, and inconsistently formatted files. Your
> system has to cope with that."

The dataset is **organized by primary category**: each file lives in
exactly one folder, picked for the characteristic it most uniquely
exercises. Many documents also stress other categories — those
cross-category contributions are shown in the matrix below.

`datasets/truepdf/` is intentionally separate from this messy-document
matrix. It is a clean control set of official U.S. Courts text-layer PDFs
for plain end-to-end checks where the expected behavior is normal PDF text
extraction, not image-based extraction.

## Per-document inspection (verified via PyMuPDF)

| File | Pages | Size | Effective DPI | Text layer | Notes |
|---|---:|---:|---:|---|---|
| `scanned-pages/ucsf_idl_xqjn0082_summons.pdf` | 4 | 149 KB | ~204 | text-layer mojibake | court summons fax scan |
| `scanned-pages/sec_vprr_02000259_form_fx.pdf` | 4 | 133 KB | ~300 | none | SEC Form F-X agent-of-process appointment |
| `scanned-pages/sec_vprr_03000005_form_d.pdf` | 6 | 281 KB | ~300 | none | SEC Form D Reg-D notice, stamped "PROCESSED" |
| `low-resolution-pdfs/synthetic_noisy_lien_notice.pdf` | 1 | 153 KB | ~129 | none | synthetic JPEG-roundtripped, skewed, with margin notes |
| `low-resolution-pdfs/loc_evening_star_dc_1900.pdf` | 1 | 1284 KB | ~150 | text-layer mojibake | LoC Chronicling America microfilm |
| `low-resolution-pdfs/sec_vprr_07020010_husqvarna.pdf` | 3 | 102 KB | ~194 | none | SEC Form 12G3-2(b) Husqvarna paper filing |
| `handwritten-notes/lee_va_chancery_1863.pdf` | 5 | 791 KB | ~150 | none | Civil-War iron-gall cursive (sliced from 13 pages) |
| `handwritten-notes/macdougall_field_notes_1899.pdf` | 5 | 358 KB | ~100 | none | Smithsonian surveyor field-notes, multiple writers |
| `handwritten-notes/marion_county_probate_orders_1833.pdf` | 5 | 578 KB | ~100 | degraded extracted text | Indiana court-clerk probate orders, antebellum cursive |
| `partially-illegible-records/ucsf_idl_rhwy0076.pdf` | 5 | 149 KB | ~204 | heavy mojibake | `T2tE`, `FAI 212AtsaiT7` |
| `partially-illegible-records/sec_vprr_09040001.pdf` | 5 | 149 KB | ~301 | broken | `UMTEDSTATES`, `SECURITIESANDEXCHANGECOMMISSION` (sliced from 16) |
| `partially-illegible-records/sec_vprr_09045744.pdf` | 5 | 332 KB | ~303 | fragmented | `Dlvi Ion of C` |
| `inconsistently-formatted-files/cuad_hosting_agreement.pdf` | 3 | 130 KB | vector | clean | short transactional contract |
| `inconsistently-formatted-files/scotus_tiktok_v_garland.pdf` | 5 | 65 KB | vector | clean | judicial slip opinion (sliced from 27) |
| `inconsistently-formatted-files/worcester_ma_probate.pdf` | 5 | 97 KB | vector | clean | spreadsheet-style probate inventory (sliced from 20) |

## Clean true-PDF control set

| File | Pages | Text layer | Notes |
|---|---:|---|---|
| `truepdf/ao085_notice_consent_magistrate.pdf` | 1 | clean | notice/consent form for magistrate judge reference |
| `truepdf/ao088b_subpoena_produce.pdf` | 3 | clean | subpoena form for documents, information, objects, or premises inspection |
| `truepdf/ao398_notice_waive_service.pdf` | 1 | clean | notice of lawsuit and request to waive service |
| `truepdf/ao399_waiver_service.pdf` | 1 | clean | waiver of service of summons |
| `truepdf/ao440_summons_civil_action.pdf` | 2 | clean | summons in a civil action |

## Cross-category coverage matrix

Primary category in **bold**. Secondary categories shown for transparency
— the dataset deliberately mixes characteristics so a single run
exercises multiple paths.

| File | Scanned | Low-res | Handwriting | Partially illegible | Inconsistent format |
|---|---|---|---|---|---|
| `scanned-pages/ucsf_idl_xqjn0082_summons.pdf` | **primary** | fax-grade | no | yes | yes |
| `scanned-pages/sec_vprr_02000259_form_fx.pdf` | **primary** | no | no | no | yes (foreign-issuer form) |
| `scanned-pages/sec_vprr_03000005_form_d.pdf` | **primary** | no | no | no | yes (SEC filing form) |
| `low-resolution-pdfs/synthetic_noisy_lien_notice.pdf` | yes (image-only) | **primary** | yes (margin notes) | yes (corrupted cert#) | yes |
| `low-resolution-pdfs/loc_evening_star_dc_1900.pdf` | yes (microfilm) | **primary** | no | yes (mojibake) | yes (newspaper) |
| `low-resolution-pdfs/sec_vprr_07020010_husqvarna.pdf` | yes (image-only) | **primary** | no | no | yes |
| `handwritten-notes/lee_va_chancery_1863.pdf` | yes (no text layer) | yes (~150 DPI) | **primary** | yes (faded ink) | yes (cursive-only) |
| `handwritten-notes/macdougall_field_notes_1899.pdf` | yes | yes (~100 DPI) | **primary** | partial | yes (red-lined journal pages, diagrams) |
| `handwritten-notes/marion_county_probate_orders_1833.pdf` | yes | yes (~100 DPI) | **primary** | partial | yes (court-clerk paragraph format) |
| `partially-illegible-records/ucsf_idl_rhwy0076.pdf` | yes (fax scan) | partial | partial (ink notations) | **primary** | yes (fax + caption + marginalia) |
| `partially-illegible-records/sec_vprr_09040001.pdf` | yes (image-only p1) | no | no | **primary** | yes |
| `partially-illegible-records/sec_vprr_09045744.pdf` | yes (paper filing) | no | no | **primary** | yes |
| `inconsistently-formatted-files/cuad_hosting_agreement.pdf` | no | no | no | no | **primary** (vector contract) |
| `inconsistently-formatted-files/scotus_tiktok_v_garland.pdf` | no | no | no | no | **primary** (judicial syllabus + concurrence) |
| `inconsistently-formatted-files/worcester_ma_probate.pdf` | no | no | partial (refs handwritten vouchers) | partial (gap fields) | **primary** (table layout) |

## Category totals (primary + secondary)

| Category | Primary docs | Also-exercised by | Total docs surfacing this characteristic |
|---|---:|---:|---:|
| Scanned pages | 3 | 9 | 12 |
| Low-resolution PDFs | 3 | 4 | 7 |
| Handwritten notes | 3 | 1 (synthetic margin notes) | 4 |
| Partially illegible records | 3 | 7 | 10 |
| Inconsistently formatted files | 3 | 11 | 14 |

## Honest assessment

- **Equal primary representation**: 3 documents per category, all sliced
  to ≤6 pages, all ≤1.3 MB. This is the contract this dataset honors.
- **Cross-coverage is heavy**: most "scanned" or "low-res" docs also
  exercise illegibility (mojibake) and format-variety, because that's
  what real messy legal documents look like.
- **Handwriting is now real**: three independent handwritten primary
  documents spanning 1833 → 1899 (Indiana probate, Virginia chancery,
  Smithsonian surveyor field notes) plus the synthetic margin notes.
  Each is a distinct period, document type, and handwriting style.
- **Low-res is verifiable**: every doc in `low-resolution-pdfs/` has a
  measured effective DPI under 200, verified by PyMuPDF
  (`page.get_images()` → `extract_image()['width']` / page width in
  points × 72).
- **Page caps are honest**: every PDF here is ≤6 pages, so a full
  multi-category run stays small even on the multimodal image-based extraction path.
- **Public eval is stricter**: `eval/public-documents/` is the downloaded
  external bundle, pinned by `eval/public_eval_manifest.json`, with 15 files across
  the same five categories and every file verified at ≤5 pages.

## Reproducing the measurements

```bash
uv run python -c "
import fitz
from pathlib import Path

for path in sorted(Path('datasets').rglob('*.pdf')):
    doc = fitz.open(path)
    page = doc[0]
    text_len = len(page.get_text())
    images = page.get_images(full=True)
    dpi = '—'
    if images:
        xref = images[0][0]
        img = doc.extract_image(xref)
        dpi = f\"{img['width'] * 72 / page.rect.width:.0f}\"
    print(f'{str(path):60s} pages={doc.page_count} p1_text={text_len} p1_dpi={dpi}')
    doc.close()
"
```
