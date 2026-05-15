// API base resolution: in dev the Vite dev server proxies API routes back to
// FastAPI, so we use same-origin empty base. In prod the FastAPI app serves the
// built UI under /ui/, and routes like /runs live at the origin root, so an
// empty base is still correct. Override via VITE_API_BASE when the backend lives
// on a different origin entirely.

export const API_BASE: string = (import.meta.env.VITE_API_BASE as string | undefined)?.trim() || "";

export function apiUrl(path: string): string {
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE}${normalized}`;
}
