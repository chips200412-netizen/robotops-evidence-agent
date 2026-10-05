"""Five-tool parameter formula fixtures; not a measured 50-case benchmark."""

from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
import json
from pathlib import Path

from app.evaluation.dataset import FrozenDataset, load_frozen_dataset


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "eval" / "parameter_formula_v1.json"
FIXTURE_SHA256 = "c3f4f1cb2c3863c686940ebefbc68eb31d0f7ba45b39e6ab38ece46fd05a636e"
_TOOLS = (
    "search_manual", "search_fault_cases", "query_device_information",
    "query_simulated_telemetry", "analyze_synthetic_ros_log",
)
_ROOT_KEYS = {
    "schema_version", "status", "provenance", "license_scope",
    "publication_status", "dataset_version", "cases_sha256",
    "corpus_sha256", "rows",
}
_ROW_KEYS = {"fixture_id", "tool_name", "public_context", "expected_arguments"}


@dataclass(frozen=True)
class ToolParameterCall:
    tool_name: str
    arguments: dict[str, str | int]


@dataclass(frozen=True)
class ToolParameterCandidate:
    calls: tuple[ToolParameterCall, ...]


@dataclass(frozen=True)
class ParameterFormulaResult:
    status: str
    fixture_id: str
    correct: bool
    dataset_version: str
    cases_sha256: str
    corpus_sha256: str
    fixture_sha256: str


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("invalid parameter formula fixture")
        value[key] = item
    return value


def _valid_arguments(tool_name: str, context: dict, arguments: dict) -> bool:
    if tool_name in _TOOLS[:2]:
        return (
            set(context) == {"device_model", "fault_category", "fault_description"}
            and all(type(value) is str and value.strip() == value and value for value in context.values())
            and set(arguments) == {"query", "top_k"}
            and type(arguments["query"]) is str
            and arguments["query"] == " ".join(context[key] for key in (
                "device_model", "fault_category", "fault_description",
            ))
            and type(arguments["top_k"]) is int and arguments["top_k"] == 5
        )
    if tool_name == "query_device_information":
        return (
            set(context) == {"device_model"}
            and type(context["device_model"]) is str and bool(context["device_model"])
            and set(arguments) == {"device_model"}
            and type(arguments["device_model"]) is str
            and arguments["device_model"] == context["device_model"]
        )
    kind = "telemetry" if tool_name == "query_simulated_telemetry" else "ros_log"
    limit_name = "max_points" if kind == "telemetry" else "max_entries"
    return (
        set(context) == {"device_model", "resource_kind"}
        and type(context["device_model"]) is str and context["device_model"] == "EduBot A1"
        and type(context["resource_kind"]) is str and context["resource_kind"] == kind
        and set(arguments) == {
            "device_model", "resource_id", "start_ms", "end_ms", limit_name, "max_bytes",
        }
        and type(arguments["device_model"]) is str
        and arguments["device_model"] == context["device_model"]
        and type(arguments["resource_id"]) is str
        and arguments["resource_id"] == (
            "telemetry-edubot-a1-v1" if kind == "telemetry" else "roslog-edubot-a1-v1"
        )
        and all(type(arguments[key]) is int for key in (
            "start_ms", "end_ms", limit_name, "max_bytes",
        ))
        and arguments["start_ms"] == 1789344000000
        and arguments["end_ms"] == 1789344600000
        and arguments[limit_name] == (120 if kind == "telemetry" else 40)
        and arguments["max_bytes"] == 32768
    )


def _load_fixture() -> tuple[tuple[dict, ...], FrozenDataset]:
    try:
        raw = FIXTURE_PATH.read_bytes()
        if len(raw) > 16_384 or sha256(raw).hexdigest() != FIXTURE_SHA256:
            raise ValueError("invalid parameter formula fixture")
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs)
        if type(value) is not dict or set(value) != _ROOT_KEYS:
            raise ValueError("invalid parameter formula fixture")
        dataset = load_frozen_dataset(ROOT / "eval")
        if (
            type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["status"] != "FORMULA_TEST_ONLY"
            or value["provenance"] != "SELF_AUTHORED_SYNTHETIC"
            or value["license_scope"] != "LOCAL_EVALUATION_ONLY"
            or value["publication_status"] != "NOT_AUTHORIZED"
            or value["dataset_version"] != dataset.version
            or value["cases_sha256"] != dataset.cases_sha256
            or value["corpus_sha256"] != dataset.corpus_sha256
            or type(value["rows"]) is not list or len(value["rows"]) != len(_TOOLS)
        ):
            raise ValueError("invalid parameter formula fixture")
        for index, row in enumerate(value["rows"], 1):
            if (
                type(row) is not dict or set(row) != _ROW_KEYS
                or row["fixture_id"] != f"PAR-{index:03d}"
                or row["tool_name"] != _TOOLS[index - 1]
                or type(row["public_context"]) is not dict
                or type(row["expected_arguments"]) is not dict
                or not _valid_arguments(
                    row["tool_name"], row["public_context"], row["expected_arguments"],
                )
            ):
                raise ValueError("invalid parameter formula fixture")
        return tuple(value["rows"]), dataset
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("invalid parameter formula fixture") from exc


def _fixture_row(fixture_id: str) -> tuple[dict, FrozenDataset]:
    if type(fixture_id) is not str:
        raise ValueError("invalid parameter formula fixture id")
    rows, dataset = _load_fixture()
    row = next((item for item in rows if item["fixture_id"] == fixture_id), None)
    if row is None:
        raise ValueError("invalid parameter formula fixture id")
    return row, dataset


def public_parameter_context(fixture_id: str) -> dict[str, object]:
    """Expose only a copied synthetic context, never the expected arguments."""
    row, _ = _fixture_row(fixture_id)
    return {
        "fixture_id": row["fixture_id"],
        "tool_name": row["tool_name"],
        "context": dict(row["public_context"]),
    }


def score_parameter_formula(
    fixture_id: str,
    candidate: ToolParameterCandidate,
) -> ParameterFormulaResult:
    """Evaluate one explicit formula candidate; never execute its tool call."""
    row, dataset = _fixture_row(fixture_id)
    if (
        type(candidate) is not ToolParameterCandidate
        or set(vars(candidate)) != {field.name for field in fields(ToolParameterCandidate)}
        or type(candidate.calls) is not tuple
        or any(
            type(call) is not ToolParameterCall
            or set(vars(call)) != {field.name for field in fields(ToolParameterCall)}
            or type(call.tool_name) is not str
            or type(call.arguments) is not dict
            or any(type(key) is not str for key in call.arguments)
            for call in candidate.calls
        )
    ):
        raise ValueError("invalid parameter formula candidate")
    correct = False
    if len(candidate.calls) == 1:
        call = candidate.calls[0]
        expected = row["expected_arguments"]
        correct = (
            call.tool_name == row["tool_name"]
            and set(call.arguments) == set(expected)
            and all(
                type(call.arguments[key]) is type(expected[key])
                and call.arguments[key] == expected[key]
                for key in expected
            )
        )
    return ParameterFormulaResult(
        status="FORMULA_TEST_ONLY", fixture_id=fixture_id, correct=correct,
        dataset_version=dataset.version,
        cases_sha256=dataset.cases_sha256,
        corpus_sha256=dataset.corpus_sha256,
        fixture_sha256=FIXTURE_SHA256,
    )
