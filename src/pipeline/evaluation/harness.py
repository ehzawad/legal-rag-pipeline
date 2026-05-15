from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from pipeline.io import write_json


@dataclass(frozen=True, slots=True)
class HarnessCase:
    case_id: str
    input_dir: str
    task: str
    deterministic_checks: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HarnessCaseResult:
    case_id: str
    output_dir: str
    deterministic: dict[str, Any]
    artifacts: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HarnessResult:
    output_dir: str
    summary: dict[str, Any]
    cases: list[HarnessCaseResult]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False, sort_keys=True)


HarnessRunner = Callable[[HarnessCase, Path], Mapping[str, Any]]


def run_harness(
    cases: list[HarnessCase],
    output_dir: Path,
    *,
    runner: HarnessRunner,
) -> HarnessResult:
    """Run repeatable harness cases and evaluate persisted deterministic facts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[HarnessCaseResult] = []
    for case in cases:
        case_dir = output_dir / case.case_id
        payload = dict(runner(case, case_dir))
        static = payload.get("static") if isinstance(payload.get("static"), Mapping) else {}
        deterministic = _score_deterministic(static, case.deterministic_checks)
        artifacts = {
            key: str(value)
            for key, value in (payload.get("artifacts") or {}).items()
            if isinstance(key, str)
        }
        result = HarnessCaseResult(
            case_id=case.case_id,
            output_dir=str(case_dir),
            deterministic=deterministic,
            artifacts=artifacts,
        )
        write_json(
            case_dir / "harness_case.json",
            {
                "case": asdict(case),
                "result": asdict(result),
                "runner": payload,
            },
        )
        results.append(result)

    summary = {
        "case_count": len(results),
        "deterministic_passed": sum(1 for result in results if result.deterministic.get("passed")),
        "passed": all(result.deterministic.get("passed") for result in results),
    }
    harness_result = HarnessResult(str(output_dir), summary, results)
    write_json(output_dir / "harness_report.json", asdict(harness_result))
    return harness_result


def _score_deterministic(static: Mapping[str, Any], checks: Mapping[str, Any]) -> dict[str, Any]:
    details: dict[str, bool] = {}
    for name, expected in checks.items():
        if name.startswith("min_"):
            metric = name.removeprefix("min_")
            details[name] = _number(_observed(static, metric)) >= float(expected)
        elif name.startswith("max_"):
            metric = name.removeprefix("max_")
            details[name] = _number(_observed(static, metric)) <= float(expected)
        elif name.startswith("equals_"):
            metric = name.removeprefix("equals_")
            details[name] = _observed(static, metric) == expected
    return {"passed": all(details.values()) if details else True, "checks": details, "observed": dict(static)}


def _number(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _observed(static: Mapping[str, Any], dotted: str) -> Any:
    aliases = {
        "unsupported_sections": "unsupported_section_count",
        "unsupported_claims": "unsupported_claim_count",
    }
    dotted = aliases.get(dotted, dotted)
    if dotted in static:
        return static.get(dotted)
    current: Any = static
    for part in dotted.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current
