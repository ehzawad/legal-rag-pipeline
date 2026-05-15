import { Suspense, lazy } from "react";
import { Route, Routes } from "react-router-dom";
import Layout from "@/components/Layout";

const DashboardPage = lazy(() => import("@/pages/DashboardPage"));
const NewRunPage = lazy(() => import("@/pages/NewRunPage"));
const RunDetailPage = lazy(() => import("@/pages/RunDetailPage"));
const RunEditPage = lazy(() => import("@/pages/RunEditPage"));
const NotFoundPage = lazy(() => import("@/pages/NotFoundPage"));

export default function App() {
  return (
    <Layout>
      <Suspense fallback={<div className="empty">Loading…</div>}>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/runs/new" element={<NewRunPage />} />
          <Route path="/runs/:caseId" element={<RunDetailPage />} />
          <Route path="/runs/:caseId/edit" element={<RunEditPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </Suspense>
    </Layout>
  );
}
