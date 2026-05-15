import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router-dom";
import { SelectedEvidenceProvider } from "@/lib/evidence-selection";
import App from "./App";
import "./styles/tokens.css";
import "./styles/app.css";

const root = document.getElementById("root");
if (!root) throw new Error("Pipeline UI: #root not found in index.html");

const basename = import.meta.env.PROD ? "/ui" : "/";

const queryClient = new QueryClient();

ReactDOM.createRoot(root).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <SelectedEvidenceProvider>
        <BrowserRouter basename={basename}>
          <App />
        </BrowserRouter>
      </SelectedEvidenceProvider>
    </QueryClientProvider>
  </React.StrictMode>,
);
