import { useMemo } from "react";
import { Link } from "react-router-dom";
import { useListRunsQuery } from "@/lib/api";
import type { RunSummary } from "@/types/api";

function formatTask(task: string) {
  if (task.length <= 110) return task;
  return `${task.slice(0, 110)}…`;
}

function isSampleOrEvalRun(run: RunSummary) {
  const value = `${run.case_id} ${run.name} ${run.output_dir}`.toLowerCase();
  return [
    "all_categories",
    "all-categories",
    "public_eval",
    "public-eval",
    "e2e",
    "cleanup",
    "/run_",
    "run_e2e",
  ].some((marker) => value.includes(marker));
}

function formatGeneratedAt(value: string | undefined) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function ReviewStats({ runs }: { runs: RunSummary[] }) {
  const readyDrafts = runs.filter((run) => run.has_draft).length;
  const needsAttention = runs.filter((run) => run.warning_count > 0).length;
  return (
    <div className="grid-3 dashboard-summary" aria-label="Review queue summary">
      <div className="metric">
        <div className="label">Cases</div>
        <div className="value">{runs.length}</div>
        <div className="delta">in the review queue</div>
      </div>
      <div className="metric">
        <div className="label">Drafts Ready</div>
        <div className="value">{readyDrafts}</div>
        <div className="delta">open and cite-check before use</div>
      </div>
      <div className="metric">
        <div className="label">Needs Attention</div>
        <div className="value">{needsAttention}</div>
        <div className="delta">has open questions or weak evidence</div>
      </div>
    </div>
  );
}

function RunRow({ run }: { run: RunSummary }) {
  const generatedAt = formatGeneratedAt(run.created_at);
  return (
    <tr>
      <td>
        <Link to={`/runs/${encodeURIComponent(run.case_id)}?output_dir=${encodeURIComponent(run.output_dir)}`}>
          <strong>{run.case_id}</strong>
        </Link>
        {generatedAt ? <div className="muted row-note">Generated {generatedAt}</div> : null}
      </td>
      <td>
        <div>{formatTask(run.task)}</div>
        {run.draft_title ? <div className="muted row-note">{run.draft_title}</div> : null}
      </td>
      <td>
        <div className="row-flex" style={{ gap: 6 }}>
          {run.warning_count > 0 ? (
            <span className="badge is-warning">Open questions</span>
          ) : (
            <span className="badge is-ok">Ready for review</span>
          )}
          {!run.has_draft ? <span className="badge is-danger">No draft</span> : null}
        </div>
      </td>
      <td>
        <div className="review-actions">
          <Link
            to={`/runs/${encodeURIComponent(run.case_id)}?output_dir=${encodeURIComponent(run.output_dir)}`}
            className="button secondary"
          >
            Open draft
          </Link>
          <Link
            to={`/runs/${encodeURIComponent(run.case_id)}/edit?output_dir=${encodeURIComponent(run.output_dir)}`}
            className="button ghost"
          >
            Teach from edit
          </Link>
        </div>
      </td>
    </tr>
  );
}

export default function DashboardPage() {
  const { data, isLoading, isError, refetch } = useListRunsQuery({});
  const runs = data?.runs ?? [];
  const reviewRuns = useMemo(
    () => runs.filter((run) => run.reviewable !== false && !isSampleOrEvalRun(run)),
    [runs],
  );
  return (
    <div>
      <div className="topbar">
        <div>
          <h2>Review queue</h2>
          <p className="topbar-sub">
            Cases ready for human legal review. Open a draft, check its citations, then teach
            the system from any edits.
          </p>
        </div>
        <div className="row-flex">
          <button type="button" className="button secondary" onClick={() => refetch()}>
            Refresh
          </button>
          <Link to="/runs/new" className="button">
            Upload documents
          </Link>
        </div>
      </div>

      <ReviewStats runs={reviewRuns} />

      <div className="card">
        <div className="section-heading-row">
          <div>
            <h3>Cases</h3>
            <p className="muted" style={{ margin: 0, fontSize: 12 }}>
              Only draft-ready case reviews appear in this queue.
            </p>
          </div>
        </div>
        {isLoading ? <div className="empty">Loading runs…</div> : null}
        {isError ? <div className="alert">Could not load runs. Is the API reachable?</div> : null}
        {data && reviewRuns.length === 0 ? (
          <div className="empty">
            No review cases yet. <Link to="/runs/new">Upload documents</Link> to start the
            first case review.
          </div>
        ) : null}
        {data && reviewRuns.length > 0 ? (
          <table className="table">
            <thead>
              <tr>
                <th>Case</th>
                <th>Review Request</th>
                <th>Review Status</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {reviewRuns.map((run) => (
                <RunRow key={`${run.case_id}-${run.output_dir}`} run={run} />
              ))}
            </tbody>
          </table>
        ) : null}
      </div>
    </div>
  );
}
