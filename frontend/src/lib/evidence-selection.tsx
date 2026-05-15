import { createContext, useContext, useMemo, useState, type ReactNode } from "react";

export interface SelectedEvidence {
  caseId: string;
  evidenceId: string;
  quote?: string;
}

interface SelectedEvidenceContextValue {
  selected: SelectedEvidence | null;
  setSelected: (next: SelectedEvidence | null) => void;
}

const SelectedEvidenceContext = createContext<SelectedEvidenceContextValue | null>(null);

export function SelectedEvidenceProvider({ children }: { children: ReactNode }) {
  const [selected, setSelected] = useState<SelectedEvidence | null>(null);
  const value = useMemo(() => ({ selected, setSelected }), [selected]);
  return (
    <SelectedEvidenceContext.Provider value={value}>{children}</SelectedEvidenceContext.Provider>
  );
}

export function useSelectedEvidence(): SelectedEvidenceContextValue {
  const ctx = useContext(SelectedEvidenceContext);
  if (!ctx) {
    throw new Error(
      "useSelectedEvidence must be used inside <SelectedEvidenceProvider>",
    );
  }
  return ctx;
}
