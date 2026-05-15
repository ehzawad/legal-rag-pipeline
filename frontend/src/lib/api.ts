import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiUrl } from "@/env";
import type {
  Draft,
  EditSubmitRequest,
  EditSubmitResponse,
  EvidenceChunk,
  ProcessedDocument,
  RunListResponse,
  RunPipelineRequest,
  RunPipelineResponse,
  RunSummary,
  UploadResponse,
} from "@/types/api";

// Keys are exported so callers (and mutation onSuccess handlers) can invalidate
// or read specific query slices without having to redeclare the tuple shape.
export const queryKeys = {
  health: ["health"] as const,
  runs: (root?: string) => ["runs", root ?? null] as const,
  runSummary: (caseId: string, outputDir?: string) =>
    ["run", caseId, "summary", outputDir ?? null] as const,
  draftJson: (caseId: string, outputDir?: string) =>
    ["run", caseId, "draft_json", outputDir ?? null] as const,
  draftMarkdown: (caseId: string, outputDir?: string) =>
    ["run", caseId, "draft_md", outputDir ?? null] as const,
  retrievedEvidence: (caseId: string, outputDir?: string) =>
    ["run", caseId, "retrieved_evidence", outputDir ?? null] as const,
  processedDocuments: (caseId: string, outputDir?: string) =>
    ["run", caseId, "processed_documents", outputDir ?? null] as const,
};

interface FetchOptions {
  method?: string;
  body?: BodyInit;
  headers?: Record<string, string>;
  signal?: AbortSignal;
  responseType?: "json" | "text";
}

// Error shape thrown by `request` so call sites can branch on `status` and
// surface server-provided detail messages (`error.data.detail`) uniformly.
export interface FetchError {
  status: number | "FETCH_ERROR";
  data?: unknown;
  error?: string;
}

async function request<T>(
  path: string,
  params?: Record<string, string | undefined> | URLSearchParams,
  options: FetchOptions = {},
): Promise<T> {
  let url = apiUrl(path);
  if (params) {
    const search = params instanceof URLSearchParams ? params : new URLSearchParams();
    if (!(params instanceof URLSearchParams)) {
      for (const [key, value] of Object.entries(params)) {
        if (value !== undefined && value !== null && value !== "") {
          search.set(key, value);
        }
      }
    }
    const query = search.toString();
    if (query) url += `?${query}`;
  }
  let response: Response;
  try {
    response = await fetch(url, {
      method: options.method ?? "GET",
      body: options.body,
      headers: options.headers,
      credentials: "omit",
      signal: options.signal,
    });
  } catch (cause) {
    const err: FetchError = { status: "FETCH_ERROR", error: String(cause) };
    throw err;
  }
  if (!response.ok) {
    let data: unknown = undefined;
    try {
      data = await response.json();
    } catch {
      try {
        data = await response.text();
      } catch {
        data = undefined;
      }
    }
    const err: FetchError = { status: response.status, data };
    throw err;
  }
  if (options.responseType === "text") {
    return (await response.text()) as unknown as T;
  }
  return (await response.json()) as T;
}

export function useHealthQuery() {
  return useQuery<{ status: string }, FetchError>({
    queryKey: queryKeys.health,
    queryFn: ({ signal }) => request<{ status: string }>("/healthz", undefined, { signal }),
    refetchInterval: 30_000,
    refetchIntervalInBackground: false,
    refetchOnWindowFocus: true,
  });
}

export function useListRunsQuery(args: { root?: string } = {}) {
  const { root } = args;
  return useQuery<RunListResponse, FetchError>({
    queryKey: queryKeys.runs(root),
    queryFn: ({ signal }) =>
      request<RunListResponse>("/runs", root ? { root } : undefined, { signal }),
  });
}

export function useGetRunSummaryQuery(args: { caseId: string; outputDir?: string }) {
  const { caseId, outputDir } = args;
  return useQuery<RunSummary, FetchError>({
    queryKey: queryKeys.runSummary(caseId, outputDir),
    queryFn: ({ signal }) =>
      request<RunSummary>(
        `/runs/${encodeURIComponent(caseId)}/summary`,
        outputDir ? { output_dir: outputDir } : undefined,
        { signal },
      ),
    enabled: caseId.length > 0,
  });
}

export function useGetDraftJsonQuery(args: { caseId: string; outputDir?: string }) {
  const { caseId, outputDir } = args;
  return useQuery<Draft, FetchError>({
    queryKey: queryKeys.draftJson(caseId, outputDir),
    queryFn: ({ signal }) =>
      request<Draft>(
        `/runs/${encodeURIComponent(caseId)}/artifacts/draft_json`,
        outputDir ? { output_dir: outputDir } : undefined,
        { signal },
      ),
    enabled: caseId.length > 0,
  });
}

export function useGetDraftMarkdownQuery(args: { caseId: string; outputDir?: string }) {
  const { caseId, outputDir } = args;
  return useQuery<string, FetchError>({
    queryKey: queryKeys.draftMarkdown(caseId, outputDir),
    queryFn: ({ signal }) =>
      request<string>(
        `/runs/${encodeURIComponent(caseId)}/artifacts/draft_md`,
        outputDir ? { output_dir: outputDir } : undefined,
        { signal, responseType: "text" },
      ),
    enabled: caseId.length > 0,
  });
}

export function useGetRetrievedEvidenceQuery(
  args: { caseId: string; outputDir?: string },
  options: { skip?: boolean } = {},
) {
  const { caseId, outputDir } = args;
  return useQuery<EvidenceChunk[], FetchError>({
    queryKey: queryKeys.retrievedEvidence(caseId, outputDir),
    queryFn: ({ signal }) =>
      request<EvidenceChunk[]>(
        `/runs/${encodeURIComponent(caseId)}/artifacts/retrieved_evidence`,
        outputDir ? { output_dir: outputDir } : undefined,
        { signal },
      ),
    enabled: caseId.length > 0 && !options.skip,
  });
}

export function useGetProcessedDocumentsQuery(
  args: { caseId: string; outputDir?: string },
  options: { skip?: boolean } = {},
) {
  const { caseId, outputDir } = args;
  return useQuery<ProcessedDocument[], FetchError>({
    queryKey: queryKeys.processedDocuments(caseId, outputDir),
    queryFn: ({ signal }) =>
      request<ProcessedDocument[]>(
        `/runs/${encodeURIComponent(caseId)}/artifacts/processed_documents`,
        outputDir ? { output_dir: outputDir } : undefined,
        { signal },
      ),
    enabled: caseId.length > 0 && !options.skip,
  });
}

export function useRunPipelineMutation() {
  const queryClient = useQueryClient();
  return useMutation<RunPipelineResponse, FetchError, RunPipelineRequest>({
    mutationFn: (body) =>
      request<RunPipelineResponse>("/runs", undefined, {
        method: "POST",
        body: JSON.stringify(body),
        headers: { "Content-Type": "application/json" },
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["runs"] });
    },
  });
}

export function useUploadDocumentsMutation() {
  return useMutation<UploadResponse, FetchError, { caseId: string; files: File[] }>({
    mutationFn: ({ caseId, files }) => {
      const data = new FormData();
      for (const file of files) {
        data.append("files", file, file.name);
      }
      return request<UploadResponse>(`/uploads/${encodeURIComponent(caseId)}`, undefined, {
        method: "POST",
        body: data,
      });
    },
  });
}

export function useSubmitEditMutation() {
  const queryClient = useQueryClient();
  return useMutation<
    EditSubmitResponse,
    FetchError,
    { caseId: string; body: EditSubmitRequest }
  >({
    mutationFn: ({ caseId, body }) =>
      request<EditSubmitResponse>(`/runs/${encodeURIComponent(caseId)}/edits`, undefined, {
        method: "POST",
        body: JSON.stringify(body),
        headers: { "Content-Type": "application/json" },
      }),
    onSuccess: (_data, variables) => {
      // Refresh the case detail surface so a fresh draft / evidence load
      // reflects any side-effects the backend applied to the run artifacts.
      queryClient.invalidateQueries({ queryKey: ["run", variables.caseId] });
    },
  });
}
