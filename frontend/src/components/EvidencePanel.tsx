import { useMemo } from "react";
import type { EvidenceChunk } from "@/types/api";
import { evidenceKind } from "@/lib/citations";

interface EvidencePanelProps {
  evidenceId: string | null;
  evidence: EvidenceChunk[];
  citationQuote?: string;
}

export default function EvidencePanel({ evidenceId, evidence, citationQuote }: EvidencePanelProps) {
  const chunk = useMemo(
    () => evidence.find((entry) => entry.evidence_id === evidenceId) ?? null,
    [evidence, evidenceId],
  );

  if (!evidenceId) {
    return (
      <aside className="evidence-panel">
        <h4 className="muted">No citation selected</h4>
        <div className="meta">
          Click any citation in the draft to see the exact quote, page, and source text
          that supports it.
        </div>
      </aside>
    );
  }
  if (!chunk) {
    return (
      <aside className="evidence-panel">
        <h4>{evidenceId}</h4>
        <div className="alert">
          This citation is not available in the saved evidence for this case.
        </div>
      </aside>
    );
  }
  const kind = evidenceKind(evidenceId);
  const metadata = chunk.metadata ?? {};
  const extractionMethod = typeof metadata["extraction_method"] === "string"
    ? (metadata["extraction_method"] as string)
    : "";
  const confidence = typeof metadata["confidence"] === "number"
    ? (metadata["confidence"] as number)
    : undefined;
  const chunkType = typeof metadata["chunk_type"] === "string"
    ? (metadata["chunk_type"] as string)
    : kind;
  const warnings = Array.isArray(metadata["warnings"]) ? (metadata["warnings"] as string[]) : [];
  return (
    <aside className="evidence-panel">
      <h4>{evidenceId}</h4>
      <div className="meta">
        <strong>{chunk.filename}</strong>
        <br />
        page {chunk.page_number || "—"}
        {chunkType ? (
          <>
            <br />
            source type: <em>{chunkType === "fields" ? "extracted field" : "page text"}</em>
          </>
        ) : null}
        {extractionMethod ? (
          <>
            <br />text extraction: <em>{extractionMethod}</em>
          </>
        ) : null}
        {typeof confidence === "number" ? (
          <>
            <br />
            extraction confidence: {(confidence * 100).toFixed(0)}%
          </>
        ) : null}
        {kind === "fields" ? (
          <>
            <br />
            <span className="badge is-warning">extracted field</span>
            <div className="muted" style={{ marginTop: 4 }}>
              This is a structured value pulled from the document. Check the page text
              before relying on it for high-stakes claims.
            </div>
          </>
        ) : (
          <>
            <br />
            <span className="badge is-ok">page text</span>
          </>
        )}
        {warnings.length ? (
          <ul style={{ margin: "6px 0 0", paddingLeft: 16 }}>
            {warnings.map((warning, index) => (
              <li key={index}>{warning}</li>
            ))}
          </ul>
        ) : null}
      </div>
      {citationQuote ? (
        <>
          <div style={{ fontSize: 11, color: "var(--color-muted)", marginBottom: 4 }}>Quote used in draft</div>
          <div className="quote">{citationQuote}</div>
        </>
      ) : null}
      <div style={{ fontSize: 11, color: "var(--color-muted)", marginBottom: 4 }}>Source text</div>
      <div className="full-text">{chunk.text}</div>
    </aside>
  );
}
