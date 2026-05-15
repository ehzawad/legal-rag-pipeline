# Per-Document Findings

## Document Quality Notes

- Handwritten and faintly-scanned pages dominate this run; treat every value carrying low extraction confidence as provisional until a paralegal eyes the original image.
- Page-level citations are required for any deadline, dollar amount, or party identification used downstream.
- Where a structured-field chunk is the only source for a key value, flag the item explicitly so the operator knows the field was derived, not transcribed from the page.

## Lien Notice (Synthetic) — provisional

- Claimant: HARBOR VIEW HOLDINGS, LLC; against NORTHSTAR MARKET LLC. [e7a7d9482447de83:fields]
- Aggregate principal: $2,850.00 for November and December 2023 access road maintenance. [e7a7d9482447de83:fields]
- Handwritten "file by 03/15?" annotation is operator-readable but not authoritative; confirm before treating as a deadline. [e7a7d9482447de83:fields]
- Notary, signature, and date fields appear to be blank on the rendered image; verify against any executed counterpart before relying on the notice.

## Website Hosting Agreement — provisional

- Parties: Natalija Tunevic / FreeCook (Client); Mitchell Vitalis / Mitchell's Web Advance, PLC (Company). [4f7f73fad7d3f00c:fields]
- Contract price $5,000; $1,900 prepayment; $3,100 on completion of scope of work. [4f7f73fad7d3f00c:fields]
- Execution date stated as January 11, 2018; project window February 8, 2018 to May 3, 2018. [4f7f73fad7d3f00c:fields]

## Deposition Renotice & Subpoena — provisional

- Renotice describes pre-examination production at least five days before deposition for the listed tobacco-related actions. [d61bba9a2aabfec3:p4:c0]
- Subpoena production date constraint: not sooner than 20 days after issuance or 15 days after service. [fd38ffa95c4f3d59:p1:c1]
- Several names and dates on the OCR-rendered pages carry replacement characters or fragment marks; do not lift party names verbatim without a clean-image check.

## SEC & Audited Report — provisional

- Form F-X identifies UFJ Shintaku Ginko Kabushiki Kaisha as filer; signed Tokyo January 22, 2002. [f292902ed5f80a01:fields]
- Annual Audited Report lists net income that the structured field renders as `716.231`; the value is suspect (likely a thousands-separator or decimal-misplacement issue) and should be reconciled against the source statements. [f30939d16de6e4bb:fields]

## Operator Directive

Use page-level citations whenever the structured field is a derived summary. Mark every section "provisional" until a human reviews the source images for the items called out above.
