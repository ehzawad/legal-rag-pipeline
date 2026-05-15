import { useEffect, useState, type FormEvent } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import {
  useGetDraftJsonQuery,
  useGetDraftMarkdownQuery,
  useGetRetrievedEvidenceQuery,
  useSubmitEditMutation,
} from "@/lib/api";
import DraftRenderer from "@/components/DraftRenderer";
import EvidencePanel from "@/components/EvidencePanel";
import { useSelectedEvidence } from "@/lib/evidence-selection";
import type { EditSubmitRequest } from "@/types/api";

type EditIntent = NonNullable<EditSubmitRequest["intent"]>;

const INTENT_OPTIONS: Array<{
  value: EditIntent;
  label: string;
  explanation: string;
  improves: string;
  placeholder: string;
  examples: string[];
}> = [
  {
    value: "auto",
    label: "Let the system route it",
    explanation:
      "Best default when you are unsure. The edit and note decide whether this is style, field, citation, rule, or rejection feedback.",
    improves:
      "Lets the system choose the right kind of lesson from your note.",
    placeholder:
      "Example: I moved uncertain dates into Open Questions because the source text is too garbled to treat them as facts.",
    examples: [
      "I moved uncertain names, dates, and amounts into Open Questions because the extracted text is too degraded to rely on them as facts.",
      "I kept only statements supported by a readable quote and removed unsupported procedural assumptions.",
    ],
  },
  {
    value: "preference",
    label: "Style or structure preference",
    explanation:
      "Use this when the facts were mostly right, but you changed section order, headings, tone, or wording.",
    improves:
      "Updates the operator profile and edit memory so future drafts follow this review style.",
    placeholder:
      "Example: Use shorter fact bullets. Keep uncertain facts in Open Questions. Do not over-explain procedural posture unless the document supports it.",
    examples: [
      "Use concise fact bullets and put uncertain extracted-text fragments in Open Questions.",
      "For probate-style records, prefer sections for Documents Reviewed, Parties, Material Facts, and Open Questions.",
    ],
  },
  {
    value: "field_correction",
    label: "Correct an extracted field",
    explanation:
      "Use this when you corrected a field value such as court, venue, governing law, date, party, or amount.",
    improves:
      "Reusable field corrections become knowledge-layer defaults; case-specific corrections stay as review guidance.",
    placeholder:
      "Example: The court is Marion County Probate Court. The extracted name fragment is uncertain and should not be treated as a party.",
    examples: [
      "The court should be Marion County Probate Court; treat the extracted name fragments as uncertain.",
      "The date fragment is not reliable. Do not treat it as an operative deadline without image review.",
    ],
  },
  {
    value: "knowledge_rule",
    label: "Standing rule for future runs",
    explanation:
      "Use this for a rule that should apply beyond this one draft, even if it is not directly stated in the source.",
    improves:
      "Writes a durable knowledge rule that future runs consult before drafting.",
    placeholder:
      "Example: For probate/order records, do not add litigation sections unless the source shows an active motion, claim, or requested relief.",
    examples: [
      "For probate/order records, do not add litigation sections unless the source shows an active motion, claim, or requested relief.",
      "Never turn garbled decimal-like extracted text into a financial amount unless the source is a financial report or the image confirms it.",
    ],
  },
  {
    value: "evidence_dispute",
    label: "Citation or evidence problem",
    explanation: "Use this when a citation is wrong, missing, weak, or points to unreadable source text.",
    improves:
      "Updates retrieval feedback and learning suggestions so future retrieval can demote weak chunks or seek better evidence.",
    placeholder:
      "Example: The Material Facts citation does not support the amount. Find a clearer page image or mark the amount unresolved.",
    examples: [
      "The Material Facts citation does not support the amount; mark the amount unresolved unless a clearer source is retrieved.",
      "The citation points to garbled extracted text. Prefer clearer page evidence for this claim.",
    ],
  },
  {
    value: "rejection",
    label: "Reject the whole draft",
    explanation: "Use this when the draft is not usable and should not become style guidance.",
    improves:
      "Records the rejection, but avoids teaching the drafter from a broad rewrite.",
    placeholder:
      "Example: Reject this draft; the extracted text is too degraded. Do not learn style from this edit.",
    examples: [
      "Reject this draft; the extracted text is too degraded. Do not learn style from this edit.",
      "This needs a fresh run with better source images before drafting.",
    ],
  },
];

function formatRequestError(error: unknown): string {
  if (!error || typeof error !== "object") return "Please try again.";
  const payload = error as { data?: { detail?: unknown } | unknown; error?: unknown };
  if (payload.data && typeof payload.data === "object" && "detail" in payload.data) {
    return String(payload.data.detail);
  }
  if (typeof payload.error === "string") return payload.error;
  return "Please try again.";
}

export default function RunEditPage() {
  const { caseId = "" } = useParams<{ caseId: string }>();
  const [params] = useSearchParams();
  const outputDir = params.get("output_dir") || undefined;
  const draftMd = useGetDraftMarkdownQuery({ caseId, outputDir });
  const draftJson = useGetDraftJsonQuery({ caseId, outputDir });
  const retrieved = useGetRetrievedEvidenceQuery({ caseId, outputDir });
  const {
    mutate: submitEdit,
    data: editResponse,
    isPending: editing,
    error: editError,
  } = useSubmitEditMutation();

  const [edited, setEdited] = useState<string>("");
  const [note, setNote] = useState<string>("");
  const [intent, setIntent] = useState<EditIntent>("auto");

  useEffect(() => {
    if (draftMd.data && !edited) {
      setEdited(draftMd.data);
    }
  }, [draftMd.data, edited]);

  const { selected } = useSelectedEvidence();

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!edited.trim()) return;
    submitEdit({
      caseId,
      body: {
        edited_text: edited,
        note,
        intent,
        category: null,
        state_dir: "state",
        output_dir: outputDir || null,
      },
    });
  }

  function learningSummary(): string {
    if (!editResponse) return "";
    const channel = editResponse.latest_event?.routing?.channel;
    const editType = editResponse.latest_classification?.classification?.edit_type;
    if (intent === "rejection" || editType === "rejection") {
      return "Rejection saved. The system will avoid learning this rewrite as a style preference.";
    }
    if (intent === "evidence_dispute" || channel === "retrieval_feedback") {
      return "Citation feedback saved. Future drafts will be steered away from weak evidence when better support is available.";
    }
    if (intent === "field_correction" || channel === "knowledge_layer") {
      return "Correction saved. Future drafts will treat the corrected fact pattern more carefully.";
    }
    if (intent === "knowledge_rule") {
      return "Standing rule saved. Future drafts will use this as drafting guidance where it fits.";
    }
    return "Edit saved. Future drafts will use the revised wording and note as review guidance.";
  }

  const intentOption = INTENT_OPTIONS.find((opt) => opt.value === intent);
  function applyNoteExample(example: string) {
    setNote((current) => {
      const trimmed = current.trim();
      return trimmed ? `${trimmed}\n${example}` : example;
    });
  }

  return (
    <div>
      <div className="topbar">
        <div>
          <h2>Edit & teach · {caseId}</h2>
          <p className="topbar-sub">
            Edit the draft, then add a short note about what you changed and why. The
            note helps the system decide what should improve on future runs.
          </p>
        </div>
        <Link
          to={`/runs/${encodeURIComponent(caseId)}${outputDir ? `?output_dir=${encodeURIComponent(outputDir)}` : ""}`}
          className="button secondary"
        >
          Back to run
        </Link>
      </div>
      <div className="split">
        <div>
          <div className="card">
            <h3>Original draft</h3>
            {draftJson.data ? (
              <DraftRenderer caseId={caseId} draft={draftJson.data} />
            ) : (
              <div className="empty">Loading draft…</div>
            )}
          </div>
          <div className="card" style={{ marginTop: 16 }}>
            <h3>Selected evidence</h3>
            <EvidencePanel
              evidenceId={selected?.caseId === caseId ? selected.evidenceId : null}
              citationQuote={selected?.caseId === caseId ? selected.quote : undefined}
              evidence={(retrieved.data ?? draftJson.data?.evidence) ?? []}
            />
          </div>
        </div>
        <form className="card" onSubmit={handleSubmit}>
          <h3>Operator edit</h3>
          <div className="notice" style={{ marginBottom: 12 }}>
            <strong>What should the note say?</strong> Write one or two plain sentences:
            what was wrong, what you changed, and whether the lesson should apply to
            future drafts. Leave legal conclusions out unless the source supports them.
          </div>
          <div className="field">
            <label htmlFor="edited">Revised draft</label>
            <textarea
              id="edited"
              value={edited}
              onChange={(e) => setEdited(e.target.value)}
              style={{ minHeight: 360, fontFamily: "var(--font-mono)", fontSize: 12 }}
            />
            <div className="muted" style={{ fontSize: 11 }}>
              Tip: keep <code>[evidence_id]</code> tokens for content that is still grounded.
              Remove a citation only when it no longer supports the edited sentence.
            </div>
          </div>
          <div className="field" style={{ marginTop: 12 }}>
            <label htmlFor="intent">What kind of edit is this?</label>
            <select id="intent" value={intent} onChange={(e) => setIntent(e.target.value as EditIntent)}>
              {INTENT_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
            {intentOption ? (
              <div className="muted" style={{ fontSize: 11 }}>
                {intentOption.explanation}
              </div>
            ) : null}
          </div>
          <div className="field" style={{ marginTop: 12 }}>
            <label htmlFor="note">Operator note</label>
            <textarea
              id="note"
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder={intentOption?.placeholder}
            />
            {intentOption ? (
              <div className="muted" style={{ fontSize: 11 }}>
                How this improves next time: {intentOption.improves}
              </div>
            ) : null}
          </div>
          {intentOption?.examples.length ? (
            <div className="example-strip" style={{ marginTop: 10 }}>
              {intentOption.examples.map((example) => (
                <button
                  key={example}
                  type="button"
                  className="note-example"
                  onClick={() => applyNoteExample(example)}
                >
                  {example}
                </button>
              ))}
            </div>
          ) : null}
          <div className="learning-map" style={{ marginTop: 12 }}>
            <div>
              <strong>Style or structure</strong>
              <span>teaches the draft shape and tone.</span>
            </div>
            <div>
              <strong>Standing rules or field fixes</strong>
              <span>become durable drafting constraints.</span>
            </div>
            <div>
              <strong>Citation problems</strong>
              <span>teach retrieval what to demote or look for next.</span>
            </div>
            <div>
              <strong>Rejections</strong>
              <span>are logged but not learned as style.</span>
            </div>
          </div>
          <div className="row-flex" style={{ marginTop: 14 }}>
            <button type="submit" className="button" disabled={editing || !edited.trim()}>
              {editing ? "Submitting…" : "Submit edit"}
            </button>
          </div>
          {editError ? (
            <div className="alert" style={{ marginTop: 12 }}>
              Could not save the edit. {formatRequestError(editError)}
            </div>
          ) : null}
          {editResponse ? (
            <div className="notice" style={{ marginTop: 12 }}>
              {learningSummary()}
            </div>
          ) : null}
        </form>
      </div>
    </div>
  );
}
