import { useHealthQuery } from "@/lib/api";

export default function HealthBadge() {
  const { data, isLoading, isError } = useHealthQuery();
  let kind = "is-warning";
  let label = "checking";
  if (isLoading) {
    kind = "";
    label = "Checking API";
  } else if (isError) {
    kind = "is-danger";
    label = "API offline";
  } else if (data?.status === "ok") {
    kind = "is-ok";
    label = "API ready";
  }
  return <div className={`badge ${kind}`}>{label}</div>;
}
