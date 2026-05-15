import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const backendOrigin = env.VITE_DEV_BACKEND_ORIGIN ?? "http://127.0.0.1:8000";
  return {
    plugins: [react()],
    base: "/ui/",
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "src"),
      },
    },
    build: {
      outDir: path.resolve(__dirname, "../src/pipeline/ui/static/dist"),
      emptyOutDir: true,
      sourcemap: false,
      target: "es2022",
    },
    server: {
      port: Number(env.VITE_DEV_PORT ?? 5173),
      host: "127.0.0.1",
      open: false,
      proxy: {
        // Forward every backend route prefix through the dev server so the
        // browser sees same-origin URLs and we avoid CORS preflights during
        // local development. The FastAPI app also accepts the configured
        // Vite origin via PIPELINE_API_CORS_ORIGINS as a fallback.
        "/healthz": backendOrigin,
        "/runs": backendOrigin,
        "/corpus": backendOrigin,
        "/index": backendOrigin,
        "/edit-memory": backendOrigin,
        "/state": backendOrigin,
        "/learn": backendOrigin,
        "/evaluate": backendOrigin,
        "/eval-suite": backendOrigin,
        "/risk": backendOrigin,
        "/ab-eval": backendOrigin,
        "/harness": backendOrigin,
        "/public-data": backendOrigin,
        "/export-dpo": backendOrigin,
      },
    },
  };
});
