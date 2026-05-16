// TypeScript projections of the pipeline schemas. Kept hand-written and narrow so
// the UI never invents fields the backend doesn't return.

export interface RunSummary {
  case_id: string;
  name: string;
  output_dir: string;
  created_at: string;
  task: string;
  draft_type: string;
  draft_title: string;
  section_count: number;
  warning_count: number;
  run_fingerprint: string;
  artifacts: Record<string, string>;
  has_draft: boolean;
  reviewable: boolean;
  run_status: string;
  failure_stage: string;
  error: string;
  has_evaluation: boolean;
  has_risk_report: boolean;
}

export interface RunListResponse {
  root: string;
  runs: RunSummary[];
}

export interface DraftSection {
  heading: string;
  body: string;
  evidence_ids: string[];
  citation_quotes: Record<string, string>;
  unsupported: boolean;
}

export interface ClaimCitation {
  evidence_id: string;
  quote: string;
  substring_grounded: boolean;
  entailed?: boolean | null;
}

export interface GroundingVerdict {
  status: string;
  reason: string;
  cited_evidence_ids: string[];
  substring_grounded_count: number;
  entailed_count?: number | null;
  action: string;
}

export interface FactClaim {
  claim_id: string;
  section_id: string;
  section: string;
  text: string;
  claim_type: string;
  confidence: string;
  citations: ClaimCitation[];
  grounding?: GroundingVerdict | null;
}

export interface CaseFactSummary {
  case_id: string;
  generated_at: string;
  title: string;
  section_order: string[];
  claims: FactClaim[];
  evidence: EvidenceChunk[];
  warnings: string[];
}

export interface EvidenceChunk {
  evidence_id: string;
  document_id: string;
  filename: string;
  page_number: number;
  text: string;
  score: number;
  metadata: Record<string, unknown>;
}

export interface Draft {
  draft_type: string;
  title: string;
  generated_at: string;
  sections: DraftSection[];
  evidence: EvidenceChunk[];
  warnings: string[];
  case_summary?: CaseFactSummary | null;
}

export interface ProcessedSource {
  path: string;
  document_id: string;
  filename: string;
  mime_type: string;
  sha256: string;
}

export interface ProcessedPage {
  document_id: string;
  page_number: number;
  text: string;
  extraction_method: string;
  confidence: number;
  warnings: string[];
}

export interface ExtractedField {
  name: string;
  value: string;
  confidence: number;
  evidence_ids: string[];
  notes: string[];
}

export interface ProcessedDocument {
  source: ProcessedSource;
  pages: ProcessedPage[];
  fields: Record<string, ExtractedField>;
  warnings: string[];
}

export interface PipelineFeaturesPayload {
  process_documents?: boolean;
  retrieve_evidence?: boolean;
  generate_draft?: boolean;
  learning_guidance?: boolean;
  operator_profile_guidance?: boolean;
  knowledge_layer_guidance?: boolean;
  exemplar_guidance?: boolean;
  retrieval_feedback?: boolean;
  field_chunks?: boolean;
  evidence_pack?: boolean;
  claim_support_check?: boolean;
  claim_entailment_judge?: boolean;
  playbook_risk?: boolean;
  max_field_chunks?: number;
  max_chunks_per_document?: number;
  field_chunk_score_penalty?: number;
}

export interface UploadedFileSummary {
  filename: string;
  size: number;
  path: string;
}

export interface UploadResponse {
  case_id: string;
  input_dir: string;
  count: number;
  files: UploadedFileSummary[];
}

export interface RunPipelineRequest {
  input_dir: string;
  output_dir: string;
  case_id: string;
  task: string;
  draft_type?: string | null;
  profile_path?: string | null;
  state_dir?: string | null;
  resume?: boolean;
  force?: boolean;
  features?: PipelineFeaturesPayload | null;
  playbook_path?: string | null;
}

export interface RunPipelineResponse {
  case_id: string;
  output_dir: string;
  artifacts: Record<string, string>;
  run_fingerprint: string;
}

export interface EditSubmitRequest {
  edited_text: string;
  note?: string;
  intent?:
    | "auto"
    | "preference"
    | "field_correction"
    | "knowledge_rule"
    | "evidence_dispute"
    | "rejection";
  actor_type?: "human" | "agent";
  actor_id?: string;
  state_dir?: string | null;
  output_dir?: string | null;
  category?: string | null;
}

export interface EditClassificationPayload {
  edit_type?: string;
  reusable_rule?: string;
  target_mechanisms?: string[];
  rule_scope?: string;
  diagnostic?: string;
  confidence?: string;
  rationale?: string;
}

export interface EditRoutingPayload {
  channel?: string;
  envelope?: string;
  target?: string;
  reason?: string;
  profile_update?: boolean;
}

export interface EditEventRecord {
  created_at?: string;
  actor?: { type?: string; id?: string };
  intent?: string;
  routing?: EditRoutingPayload;
  classification?: EditClassificationPayload;
  draft_path?: string;
  edited_path?: string;
  [extra: string]: unknown;
}

export interface EditClassificationRecord {
  timestamp?: string;
  edit_signature?: string;
  classification?: EditClassificationPayload;
  draft_path?: string;
  edited_path?: string;
}

export interface OperatorProfile {
  version: number;
  examples: unknown[];
  signals?: Record<string, unknown>;
  [extra: string]: unknown;
}

export interface EditSubmitResponse {
  case_id: string;
  run_dir: string;
  edited_path: string;
  profile_path: string;
  state_dir: string;
  profile: OperatorProfile;
  latest_event?: EditEventRecord | null;
  latest_classification?: EditClassificationRecord | null;
  intent: string;
  category?: string | null;
  edit_memory_delta: {
    patterns_before: number;
    patterns_after: number;
    patterns_added: number;
  };
}
