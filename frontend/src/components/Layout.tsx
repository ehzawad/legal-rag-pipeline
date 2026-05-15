import { type ReactNode } from "react";
import { NavLink } from "react-router-dom";
import HealthBadge from "./HealthBadge";

interface LayoutProps {
  children: ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  return (
    <div className="layout">
      <aside className="sidebar">
        <h1>
          <span className="brand-dot" aria-hidden /> Legal Draft Review
        </h1>
        <HealthBadge />
        <nav>
          <span className="nav-section">Reviews</span>
          <NavLink to="/" end>
            Dashboard
          </NavLink>
          <NavLink to="/runs/new">Upload documents</NavLink>
        </nav>
        <div className="meta">
          Grounded legal-document drafts for human review. Operator edits are captured
          and reused to improve later drafts.
        </div>
      </aside>
      <main className="main">{children}</main>
    </div>
  );
}
