import { useMemo, useRef, useState, type ChangeEvent, type DragEvent, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { useRunPipelineMutation, useUploadDocumentsMutation } from "@/lib/api";

const ALLOWED_EXTENSIONS = [".pdf", ".png", ".jpg", ".jpeg", ".txt"];
const ALLOWED_MIME = ".pdf,.png,.jpg,.jpeg,.txt";
const MAX_BYTES = 30 * 1024 * 1024;

const DEFAULT_TASK =
  "Review the supplied legal-style documents and produce a first-pass operator summary. " +
  "Adapt the sections to the document type, and include parties, dates, amounts, signatures, " +
  "procedural posture, relief, or other legal facts only when the evidence supports them.";

function suggestCaseId(): string {
  const now = new Date();
  const stamp = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, "0")}${String(
    now.getDate(),
  ).padStart(2, "0")}-${String(now.getHours()).padStart(2, "0")}${String(now.getMinutes()).padStart(2, "0")}`;
  return `case-${stamp}`;
}

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(2)} MB`;
}

function isAllowedFile(file: File): boolean {
  const lower = file.name.toLowerCase();
  return ALLOWED_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

function formatRequestError(error: unknown): string {
  if (!error || typeof error !== "object") return "Please try again.";
  const payload = error as { data?: { detail?: unknown } | unknown; error?: unknown };
  if (payload.data && typeof payload.data === "object" && "detail" in payload.data) {
    return String(payload.data.detail);
  }
  if (typeof payload.error === "string") return payload.error;
  return "Please try again.";
}

export default function NewRunPage() {
  const [files, setFiles] = useState<File[]>([]);
  const [caseId, setCaseId] = useState(suggestCaseId);
  const [task, setTask] = useState(DEFAULT_TASK);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const { mutateAsync: uploadDocs, isPending: isUploading } = useUploadDocumentsMutation();
  const { mutateAsync: runPipeline, isPending: isRunning } = useRunPipelineMutation();
  const navigate = useNavigate();

  const totalSize = useMemo(() => files.reduce((acc, f) => acc + f.size, 0), [files]);
  const submitting = isUploading || isRunning;

  function ingestFiles(incoming: FileList | File[]) {
    const list = Array.from(incoming);
    const rejected: string[] = [];
    const accepted: File[] = [];
    for (const file of list) {
      if (!isAllowedFile(file)) {
        rejected.push(`${file.name} (unsupported type)`);
        continue;
      }
      if (file.size > MAX_BYTES) {
        rejected.push(`${file.name} (${formatBytes(file.size)} > 30 MB)`);
        continue;
      }
      accepted.push(file);
    }
    setFiles((previous) => {
      const seen = new Set(previous.map((f) => `${f.name}:${f.size}`));
      const merged = previous.slice();
      for (const f of accepted) {
        const key = `${f.name}:${f.size}`;
        if (!seen.has(key)) {
          merged.push(f);
          seen.add(key);
        }
      }
      return merged;
    });
    if (rejected.length) {
      setErrorMessage(`Skipped: ${rejected.join(", ")}`);
    } else {
      setErrorMessage(null);
    }
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragOver(false);
    if (event.dataTransfer?.files?.length) {
      ingestFiles(event.dataTransfer.files);
    }
  }

  function onSelect(event: ChangeEvent<HTMLInputElement>) {
    if (event.target.files?.length) ingestFiles(event.target.files);
    event.target.value = "";
  }

  function removeFile(index: number) {
    setFiles((previous) => previous.filter((_, i) => i !== index));
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setErrorMessage(null);
    if (files.length === 0) {
      setErrorMessage("Add at least one PDF, image, or text file before running.");
      return;
    }
    let uploadResult;
    try {
      uploadResult = await uploadDocs({ caseId, files });
    } catch (error) {
      setErrorMessage(`Upload failed. ${formatRequestError(error)}`);
      return;
    }
    const inputDir = uploadResult.input_dir;
    if (!inputDir) {
      setErrorMessage("Upload returned no input_dir; cannot start run.");
      return;
    }
    const outputDir = `outputs/${caseId}`;
    let runResult;
    try {
      runResult = await runPipeline({
        input_dir: inputDir,
        output_dir: outputDir,
        case_id: caseId,
        task,
        state_dir: "state",
        force: true,
        features: {
          learning_guidance: true,
          retrieval_feedback: true,
          field_chunks: true,
        },
      });
    } catch (error) {
      setErrorMessage(`Could not prepare the draft. ${formatRequestError(error)}`);
      return;
    }
    navigate(
      `/runs/${encodeURIComponent(runResult.case_id)}?output_dir=${encodeURIComponent(runResult.output_dir)}`,
    );
  }

  return (
    <div>
      <div className="topbar">
        <div>
          <h2>Start a new review</h2>
          <p className="topbar-sub">
            Drop your legal-style documents below. The system reads them, finds supporting
            evidence, and prepares a first-pass draft for review.
          </p>
        </div>
      </div>

      <form className="card" onSubmit={handleSubmit}>
        <h3 className="is-quiet">1 · Upload source documents</h3>
        <div
          className={`dropzone${isDragOver ? " is-active" : ""}`}
          onDragOver={(event) => {
            event.preventDefault();
            setIsDragOver(true);
          }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={onDrop}
          onClick={() => fileInputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              fileInputRef.current?.click();
            }
          }}
        >
          <div className="dropzone-title">Drop files here or click to browse</div>
          <div className="muted" style={{ fontSize: 12 }}>
            Accepted: PDF, PNG, JPG/JPEG, TXT · up to 30 MB each
          </div>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept={ALLOWED_MIME}
            onChange={onSelect}
            style={{ display: "none" }}
          />
        </div>

        {files.length > 0 ? (
          <div style={{ marginTop: 12 }}>
            <div className="muted" style={{ fontSize: 12, marginBottom: 6 }}>
              {files.length} file{files.length === 1 ? "" : "s"} queued · {formatBytes(totalSize)} total
            </div>
            <ul className="file-list">
              {files.map((file, index) => (
                <li key={`${file.name}:${index}`}>
                  <span>
                    <code>{file.name}</code>{" "}
                    <span className="muted">· {formatBytes(file.size)}</span>
                  </span>
                  <button
                    type="button"
                    className="button ghost"
                    onClick={() => removeFile(index)}
                    aria-label={`Remove ${file.name}`}
                  >
                    remove
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        <h3 className="is-quiet" style={{ marginTop: 22 }}>
          2 · Describe the case
        </h3>
        <div className="form-row">
          <div className="field">
            <label htmlFor="case_id">Case id</label>
            <input
              id="case_id"
              value={caseId}
              onChange={(event) => setCaseId(event.target.value)}
              pattern="[A-Za-z0-9][A-Za-z0-9_\-.]*"
              required
            />
            <div className="muted" style={{ fontSize: 11 }}>
              Use a short internal name, for example <code>smith-probate-01</code>.
            </div>
          </div>
        </div>
        <div className="field" style={{ marginTop: 12 }}>
          <label htmlFor="task">Drafting task</label>
          <textarea
            id="task"
            value={task}
            onChange={(event) => setTask(event.target.value)}
            required
          />
        </div>

        <h3 className="is-quiet" style={{ marginTop: 22 }}>
          3 · Start review
        </h3>
        <button type="submit" className="button" disabled={submitting}>
          {isUploading ? "Uploading documents…" : isRunning ? "Preparing draft…" : "Start review"}
        </button>

        {errorMessage ? (
          <div className="alert" style={{ marginTop: 14 }}>
            {errorMessage}
          </div>
        ) : null}
        {submitting ? (
          <div className="notice" style={{ marginTop: 14 }}>
            This uses the configured extraction, retrieval, and drafting providers. Expect
            30-120 seconds depending on document count.
          </div>
        ) : null}
      </form>
    </div>
  );
}
