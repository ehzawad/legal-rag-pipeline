import { useState, useMemo } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import {
  useGetDraftJsonQuery,
  useGetProcessedDocumentsQuery,
  useGetRetrievedEvidenceQuery,
  useGetRunSummaryQuery,
} from "@/lib/api";
import { useSelectedEvidence } from "@/lib/evidence-selection";
import DraftRenderer from "@/components/DraftRenderer";
import EvidencePanel from "@/components/EvidencePanel";

type Tab = "draft" | "evidence" | "processed" | "grounding";

const TABS: Array<[Tab, string]> = [
  ["draft", "Draft"],
  ["processed", "Source documents"],
  ["evidence", "Citations"],
  ["grounding", "Grounding check"],
];

function groundingStatusLabel(status?: string | null): string {
  switch (status) {
    case "entailed":
    case "supported":
      return "Supported";
    case "scaffolding":
      return "Review note";
    case "unsupported":
    case "ungrounded":
      return "Needs review";
    default:
      return "Not checked";
  }
}

function groundingStatusClass(status?: string | null): string {
  switch (status) {
    case "entailed":
    case "supported":
      return "is-ok";
    case "unsupported":
    case "ungrounded":
      return "is-danger";
    default:
      return "is-warning";
  }
}

export default function RunDetailPage() {
  const { caseId = "" } = useParams<{ caseId: string }>();
  const [searchParams] = useSearchParams();
  const outputDir = searchParams.get("output_dir") || undefined;
  const [tab, setTab] = useState<Tab>("draft");

  const summary = useGetRunSummaryQuery({ caseId, outputDir });
  const draft = useGetDraftJsonQuery({ caseId, outputDir });
  const retrieved = useGetRetrievedEvidenceQuery(
    { caseId, outputDir },
    { skip: tab !== "evidence" && tab !== "draft" },
  );
  const processed = useGetProcessedDocumentsQuery(
    { caseId, outputDir },
    { skip: tab !== "processed" },
  );

  const { selected } = useSelectedEvidence();
  const evidenceForSelected = useMemo(() => {
    if (!selected || selected.caseId !== caseId) return null;
    return selected;
  }, [selected, caseId]);

  return (
    <div>
      <div className="topbar">
        <div>
          <h2>Case · {caseId}</h2>
          <p className="topbar-sub">
            {summary.data?.task || "Loading…"}
          </p>
        </div>
        <div className="row-flex">
          <Link
            to={`/runs/${encodeURIComponent(caseId)}/edit${outputDir ? `?output_dir=${encodeURIComponent(outputDir)}` : ""}`}
            className="button"
          >
            Edit & teach
          </Link>
          <Link to="/" className="button secondary">
            Back to queue
          </Link>
        </div>
      </div>

      <div className="tabs">
        {TABS.map(([key, label]) => (
          <button
            key={key}
            type="button"
            className={tab === key ? "is-active" : ""}
            onClick={() => setTab(key)}
          >
            {label}
          </button>
        ))}
      </div>

      {tab === "draft" ? (
        <div className="draft-layout">
          <div>
            {draft.isLoading ? <div className="empty">Loading draft…</div> : null}
            {draft.isError ? (
              <div className="alert">Could not load the draft for this case.</div>
            ) : null}
            {draft.data ? <DraftRenderer caseId={caseId} draft={draft.data} /> : null}
          </div>
          <EvidencePanel
            evidenceId={evidenceForSelected?.evidenceId ?? null}
            citationQuote={evidenceForSelected?.quote}
            evidence={(retrieved.data ?? draft.data?.evidence) ?? []}
          />
        </div>
      ) : null}

      {tab === "evidence" ? (
        <div className="card">
          <h3>Citations ({retrieved.data?.length ?? 0})</h3>
          {retrieved.isLoading ? <div className="empty">Loading citations…</div> : null}
          {retrieved.isError ? (
            <div className="alert">Could not load citations.</div>
          ) : null}
          {retrieved.data ? (
            <table className="table">
              <thead>
                <tr>
                  <th>Citation</th>
                  <th>Source</th>
                  <th>Page</th>
                  <th>Excerpt</th>
                </tr>
              </thead>
              <tbody>
                {retrieved.data.map((chunk) => (
                  <tr key={chunk.evidence_id}>
                    <td>
                      <code>{chunk.evidence_id}</code>
                    </td>
                    <td>{chunk.filename}</td>
                    <td>{chunk.page_number || "—"}</td>
                    <td style={{ maxWidth: 540 }}>
                      <span style={{ whiteSpace: "pre-wrap" }}>{chunk.text.slice(0, 240)}{chunk.text.length > 240 ? "…" : ""}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
        </div>
      ) : null}

      {tab === "processed" ? (
        <div className="card">
          <h3>Source documents ({processed.data?.length ?? 0})</h3>
          {processed.isLoading ? <div className="empty">Loading source documents…</div> : null}
          {processed.isError ? (
            <div className="alert">Could not load source document details.</div>
          ) : null}
          {processed.data?.map((doc) => (
            <details key={doc.source.document_id} className="disclosure">
              <summary>
                {doc.source.filename}{" "}
                <span className="muted" style={{ fontWeight: 400 }}>
                  · {doc.pages.length} page{doc.pages.length === 1 ? "" : "s"} ·{" "}
                  {Object.keys(doc.fields).length} fields
                </span>
              </summary>
              <div style={{ marginTop: 10 }}>
                {doc.warnings.length ? (
                  <div className="warning-list">
                    <strong>Document warnings:</strong>
                    <ul style={{ margin: "4px 0 0", paddingLeft: 18 }}>
                      {doc.warnings.map((w, i) => <li key={i}>{w}</li>)}
                    </ul>
                  </div>
                ) : null}
                <table className="table" style={{ marginBottom: 10 }}>
                  <thead>
                    <tr>
                      <th>Field</th>
                      <th>Value</th>
                      <th>Confidence</th>
                      <th>Evidence ids</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(doc.fields).map(([name, field]) => (
                      <tr key={name}>
                        <td><code>{name}</code></td>
                        <td style={{ maxWidth: 420 }}>{field.value}</td>
                        <td>{(field.confidence * 100).toFixed(0)}%</td>
                        <td>
                          {field.evidence_ids.map((id) => <code key={id} style={{ marginRight: 4 }}>{id}</code>)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <details>
                  <summary>Per-page text and confidence</summary>
                  <table className="table" style={{ marginTop: 10 }}>
                    <thead>
                      <tr>
                        <th>Page</th>
                        <th>Method</th>
                        <th>Confidence</th>
                        <th>Warnings</th>
                      </tr>
                    </thead>
                    <tbody>
                      {doc.pages.map((page) => (
                        <tr key={page.page_number}>
                          <td>{page.page_number}</td>
                          <td><code>{page.extraction_method}</code></td>
                          <td>{(page.confidence * 100).toFixed(0)}%</td>
                          <td>{page.warnings.join("; ")}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </details>
              </div>
            </details>
          ))}
        </div>
      ) : null}

      {tab === "grounding" ? (
        <div className="card">
          <h3>Grounding check</h3>
          {draft.isLoading ? <div className="empty">Loading…</div> : null}
          {draft.data?.case_summary ? (
            <>
              <div className="grid-3" style={{ marginBottom: 12 }}>
                <div className="metric">
                  <div className="label">Total claims</div>
                  <div className="value">{draft.data.case_summary.claims.length}</div>
                </div>
                <div className="metric">
                  <div className="label">Substring-grounded citations</div>
                  <div className="value">
                    {draft.data.case_summary.claims.reduce(
                      (acc, claim) =>
                        acc + claim.citations.filter((c) => c.substring_grounded).length,
                      0,
                    )}
                  </div>
                </div>
                <div className="metric">
                  <div className="label">Needs review</div>
                  <div className="value">
                    {draft.data.case_summary.claims.filter(
                      (c) =>
                        c.grounding?.status === "unsupported" ||
                        c.grounding?.status === "ungrounded",
                    ).length}
                  </div>
                </div>
              </div>
              <table className="table">
                <thead>
                  <tr>
                    <th>Section</th>
                    <th>Claim</th>
                    <th>Status</th>
                    <th>Citations</th>
                  </tr>
                </thead>
                <tbody>
                  {draft.data.case_summary.claims.map((claim) => (
                    <tr key={claim.claim_id}>
                      <td>{claim.section}</td>
                      <td>{claim.text}</td>
                      <td>
                        <span
                          className={`badge ${groundingStatusClass(claim.grounding?.status)}`}
                          title={claim.grounding?.status || "unscored"}
                        >
                          {groundingStatusLabel(claim.grounding?.status)}
                        </span>
                      </td>
                      <td>
                        {claim.citations.map((c) => (
                          <div key={`${c.evidence_id}:${c.quote}`}>
                            <code>{c.evidence_id}</code>{" "}
                            {c.substring_grounded ? (
                              <span className="badge is-ok">verbatim</span>
                            ) : (
                              <span className="badge is-warning">unmatched</span>
                            )}
                          </div>
                        ))}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          ) : (
            <div className="muted">No case-fact summary persisted for this run.</div>
          )}
        </div>
      ) : null}
    </div>
  );
}
