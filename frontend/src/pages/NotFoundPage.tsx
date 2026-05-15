import { Link } from "react-router-dom";

export default function NotFoundPage() {
  return (
    <div className="card">
      <h3>Page not found</h3>
      <p className="muted">
        Try the <Link to="/">review queue</Link>, or upload documents to start a new case review.
      </p>
    </div>
  );
}
