import { useMemo } from "react";
import { useSelectedEvidence } from "@/lib/evidence-selection";
import { tokenizeBody } from "@/lib/citations";
import type { Draft } from "@/types/api";

interface DraftRendererProps {
  caseId: string;
  draft: Draft;
}

function formatReviewFlag(warning: string): string {
  const supportMatch = warning.match(
    /^Advisory claim-support check flagged section '([^']+)'/,
  );
  if (supportMatch) {
    return `Citation check: ${supportMatch[1]} has wording that does not clearly match the quoted source text. Review the cited evidence before relying on that section.`;
  }
  if (warning.startsWith("Input processing produced warnings")) {
    return "Some source text is uncertain. Review Open Questions and Source Documents before relying on the draft.";
  }
  if (warning.startsWith("Marked section ")) {
    return "Citation check: a draft section contains factual wording that needs citation review before relying on it.";
  }
  return warning;
}

function isIrrelevantDecimalFinancialWarning(warning: string, draft: Draft): boolean {
  const warningText = warning.toLowerCase();
  if (!warningText.includes("decimal punctuation")) return false;
  if (
    !["net-income", "net income", "aggregate-revenue", "aggregate revenue", "millions"].some((term) =>
      warningText.includes(term),
    )
  ) {
    return false;
  }
  const evidenceText = draft.evidence
    .map((chunk) => {
      const metadataText = Object.values(chunk.metadata ?? {})
        .filter((value): value is string => typeof value === "string")
        .join(" ");
      return `${chunk.text} ${chunk.filename} ${chunk.document_id} ${metadataText}`;
    })
    .join(" ")
    .toLowerCase();
  return ![
    "net income",
    "net-income",
    "aggregate revenue",
    "aggregate-revenue",
    "total revenue",
    "annual audited",
    "financial statement",
    "broker-dealer",
    "form x-17a-5",
    "millions",
  ].some((term) => evidenceText.includes(term));
}

function sectionReviewLabel(heading: string): string {
  return heading.toLowerCase() === "open questions" ? "Review note" : "Needs review";
}

export default function DraftRenderer({ caseId, draft }: DraftRendererProps) {
  const { selected, setSelected } = useSelectedEvidence();
  const reviewFlags = useMemo(
    () =>
      draft.warnings
        .filter((warning) => !isIrrelevantDecimalFinancialWarning(warning, draft))
        .map((warning) => ({ display: formatReviewFlag(warning), raw: warning })),
    [draft],
  );

  // The Python drafter strips any citation whose verbatim quote fails
  // substring validation, so surviving `evidence_ids` (and the matching
  // entry in `citation_quotes`) are the authoritative grounding signal.
  // We don't try to second-guess that here; instead, we surface "absent
  // from this run's persisted evidence" as a soft hint only — the chip's
  // accessible label and tooltip explain the difference. We never show a
  // chip as "ungrounded" purely because the id isn't in `draft.evidence`,
  // because the drafter persists only a top-k slice.
  const knownEvidenceIds = useMemo(() => {
    const ids = new Set<string>();
    for (const entry of draft.evidence) {
      ids.add(entry.evidence_id);
    }
    return ids;
  }, [draft.evidence]);

  return (
    <div>
      <h2 style={{ margin: "0 0 6px" }}>{draft.title || "Draft"}</h2>
      <div className="muted" style={{ marginBottom: 14, fontSize: 12 }}>
        Generated {new Date(draft.generated_at).toLocaleString()}
      </div>
      {reviewFlags.length ? (
        <div className="warning-list">
          <strong>{reviewFlags.length} review flag{reviewFlags.length === 1 ? "" : "s"}:</strong>
          <ul style={{ margin: "4px 0 0", paddingLeft: 18 }}>
            {reviewFlags.map((warning, index) => (
              <li key={index} title={warning.raw}>
                {warning.display}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {draft.sections.map((section, sectionIndex) => {
        const tokens = tokenizeBody(section.body);
        return (
          <article
            key={`${section.heading}-${sectionIndex}`}
            className={`draft-section${section.unsupported ? " is-unsupported" : ""}`}
          >
            <h3>
              {section.heading}
              {section.unsupported ? (
                <span className="badge is-warning" style={{ marginLeft: 8 }}>
                  {sectionReviewLabel(section.heading)}
                </span>
              ) : null}
            </h3>
            <div className="body">
              {tokens.map((token, index) => {
                if (token.kind === "text") {
                  return <span key={index}>{token.text}</span>;
                }
                const id = token.evidenceId ?? "";
                const isActive = selected?.evidenceId === id && selected.caseId === caseId;
                const quote = section.citation_quotes[id];
                // "Grounded" here means the drafter persisted both a verbatim
                // quote (in citation_quotes) and the id in evidence_ids.
                // Anything else is shown neutrally — we never claim a citation
                // is ungrounded purely because the persisted top-k slice
                // omitted the chunk.
                const isGrounded = Boolean(quote) && section.evidence_ids.includes(id);
                const isMissingFromRetrieved = !knownEvidenceIds.has(id);
                const className = ["evidence-chip"];
                if (isActive) className.push("is-active");
                if (!isGrounded) className.push("is-unsupported");
                const labelGrounding = isGrounded
                  ? `verbatim quote: ${quote}`
                  : isMissingFromRetrieved
                    ? `evidence id ${id} is not present in the persisted retrieved set; click to inspect`
                    : `no verbatim quote recorded for ${id}; click to inspect`;
                return (
                  <button
                    key={index}
                    type="button"
                    className={className.join(" ")}
                    aria-label={`Show evidence ${id}: ${labelGrounding}`}
                    title={labelGrounding}
                    onClick={() =>
                      setSelected({
                        caseId,
                        evidenceId: id,
                        quote,
                      })
                    }
                  >
                    {token.text}
                  </button>
                );
              })}
            </div>
            {section.evidence_ids.length ? (
              <details className="disclosure" style={{ marginTop: 10 }}>
                <summary>Cited evidence ({section.evidence_ids.length})</summary>
                <ul style={{ margin: "8px 0 0", paddingLeft: 18 }}>
                  {section.evidence_ids.map((id) => (
                    <li key={id} style={{ marginBottom: 4 }}>
                      <code>{id}</code>
                      {section.citation_quotes[id] ? (
                        <>
                          {" — "}
                          <em>"{section.citation_quotes[id]}"</em>
                        </>
                      ) : null}
                    </li>
                  ))}
                </ul>
              </details>
            ) : null}
          </article>
        );
      })}
    </div>
  );
}
