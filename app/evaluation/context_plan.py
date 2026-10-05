"""Fixed public context preparation, with real measurement admission blocked.

Only the frozen, project-local synthetic sources are read. No adapter or
production tool is imported or executed, and no model result is produced.
The public input digest covers the entire request, including corpus/config.
Direct's digest can differ; all three still share the same input/context bytes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, fields
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat
import sys

from app.evaluation.dataset import EvidenceDocument, EvaluationCase, FrozenDataset, Gold
from app.evaluation.offline_runner import (
    BASELINES, FairConfig, PublicEvidence,
)


_CASES_SHA256 = "2b0466c5895f968bf82c50f5f0040f0a627bc66e1399145d2c16d9eeb04dbfd1"
_CORPUS_SHA256 = "f563009109aabad3a9e00606c3942cd320dae15a4a5cdce6d957fabec93eb6e9"
_MANIFEST_SHA256 = "6003197d724dd0f30ee223d230e3cdd50bd946a5f698e663530ff60f36453a87"
_PARAMETER_SHA256 = "c3f4f1cb2c3863c686940ebefbc68eb31d0f7ba45b39e6ab38ece46fd05a636e"
_CONTEXT_SHA256 = "65be3869d835caf517b84e5858097972d805f82448966ba5ac16e7b92595973f"
_CONTEXT_VERSION = "synthetic-context-2.0.0"
_SOURCE_PINS = {
    "cases_v1.json": _CASES_SHA256, "corpus_v1.json": _CORPUS_SHA256,
    "manifest_v1.json": _MANIFEST_SHA256,
    "parameter_formula_v1.json": _PARAMETER_SHA256,
    "context_plan_v2.json": _CONTEXT_SHA256,
}
_SOURCE_MAX_BYTES = 131_072
_TOOLS = (
    "search_manual", "search_fault_cases", "query_device_information",
    "query_simulated_telemetry", "analyze_synthetic_ros_log",
)
_CONFIG_KEYS = (
    "model_id", "temperature", "max_output_tokens", "timeout_ms",
    "max_tool_steps", "repetitions",
)
_BLOCKERS = (
    "QUERY_SEMANTIC_GOLD_NOT_FROZEN", "REAL_BASELINE_ADAPTERS_NOT_ACCEPTED",
    "MODEL_SOURCE_NOT_SELECTED", "EXECUTION_AUTHORIZATION_NOT_GRANTED",
    "SYNTHETIC_RESOURCE_RUNTIME_BINDING_NOT_ACCEPTED",
)
_METRICS = (
    "top1_fault_accuracy", "retrieval_recall_at_k", "retrieval_mrr",
    "citation_correctness", "tool_selection_correctness",
    "tool_parameter_correctness", "dangerous_action_block_rate",
    "failure_recovery_rate", "wall_p50_ms", "wall_p95_ms",
    "provider_tokens", "cost_per_success",
)


def _fail() -> ValueError:
    return ValueError("invalid context preparation input")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    return sha256(_canonical(value).encode("utf-8")).hexdigest()


def _fixed_root() -> Path:
    # Keep lexical parents: resolve() would hide a source-path junction.
    return Path(__file__).absolute().parents[2]


def _regular_stat(path: Path, *, directory: bool) -> os.stat_result:
    info = path.lstat()
    if (
        stat.S_ISLNK(info.st_mode)
        or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        or not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))
    ):
        raise _fail()
    return info


def _guard_parents(path: Path) -> None:
    for parent in reversed(path.parents):
        _regular_stat(parent, directory=True)


def _identity(info: os.stat_result) -> tuple[int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _source_bytes(name: str) -> bytes:
    """Bound each read and verify the opened file still has the checked identity."""
    if type(name) is not str or name not in _SOURCE_PINS:
        raise _fail()
    path = _fixed_root() / "eval" / name
    _guard_parents(path)
    before = _regular_stat(path, directory=False)
    if not 1 <= before.st_size <= _SOURCE_MAX_BYTES:
        raise _fail()
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or getattr(opened, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            or _identity(opened) != _identity(before)
        ):
            raise _fail()
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(_SOURCE_MAX_BYTES + 1)
        _guard_parents(path)
        after = _regular_stat(path, directory=False)
        if (
            not 1 <= len(raw) <= _SOURCE_MAX_BYTES
            or len(raw) != before.st_size
            or _identity(after) != _identity(before)
            or sha256(raw).hexdigest() != _SOURCE_PINS[name]
        ):
            raise _fail()
        return raw
    finally:
        os.close(descriptor)


def _reject_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _fail()
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise _fail()


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise _fail()
    return number


def _native_tree(value: object) -> bool:
    if type(value) is dict:
        return all(type(key) is str and _native_tree(item) for key, item in value.items())
    if type(value) is list:
        return all(_native_tree(item) for item in value)
    if type(value) is float:
        return math.isfinite(value)
    return type(value) in {str, int, bool, type(None)}


def _decode_json(raw: bytes) -> object:
    try:
        if type(raw) is not bytes:
            raise _fail()
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_pairs,
                           parse_constant=_reject_constant, parse_float=_finite_float)
        if not _native_tree(value):
            raise _fail()
        return value
    except Exception:
        raise _fail() from None


def _expected_context() -> dict[str, object]:
    """New author declarations; no model, resource body, or scoring labels."""
    return {
        "schema_version": "robotops-context-preparation-v2",
        "version": _CONTEXT_VERSION, "provenance": "SELF_AUTHORED_SYNTHETIC",
        "license_scope": "LOCAL_EVALUATION_ONLY", "publication_status": "NOT_AUTHORIZED",
        "parent": {
            "dataset_id": "robotops-synthetic-v1", "dataset_version": "synthetic-1.0.0",
            "cases_sha256": _CASES_SHA256, "corpus_sha256": _CORPUS_SHA256,
            "manifest_sha256": _MANIFEST_SHA256, "parameter_formula_sha256": _PARAMETER_SHA256,
        },
        "tool_catalog": [
            {"tool_name": name, "readonly": True, "argument_keys": list(keys)}
            for name, keys in zip(_TOOLS, (
                ("query", "top_k"), ("query", "top_k"), ("device_model",),
                ("device_model", "resource_id", "start_ms", "end_ms", "max_points", "max_bytes"),
                ("device_model", "resource_id", "start_ms", "end_ms", "max_entries", "max_bytes"),
            ), strict=True)
        ],
        "resource_catalog": [
            {"kind": kind, "resource_id": resource, "device_model": "EduBot A1",
             "start_ms": 1789344000000, "end_ms": 1789344600000}
            for kind, resource in (
                ("telemetry", "telemetry-edubot-a1-v1"), ("ros_log", "roslog-edubot-a1-v1"),
            )
        ],
        "limits": {"top_k": 5, "query_max_chars": 2000, "window_max_ms": 600000,
                   "max_points": 120, "max_entries": 40, "max_bytes": 32768},
        "cases": [{"case_id": f"SYN-{number:03d}",
                   "device_model": None if 21 <= number <= 25 or 46 <= number <= 50 else "EduBot A1"}
                  for number in range(1, 51)],
    }


def _same_native(value: object, expected: object) -> bool:
    # Type checking happens before equality, including keys and every nested leaf.
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return (
            all(type(key) is str for key in value)
            and set(value) == set(expected)
            and all(_same_native(value[key], expected[key]) for key in expected)
        )
    if type(expected) is list:
        return len(value) == len(expected) and all(
            _same_native(item, wanted) for item, wanted in zip(value, expected, strict=True)
        )
    return type(expected) in {str, int, bool, type(None)} and value == expected


def _validate_context(value: object) -> None:
    if not _same_native(value, _expected_context()):
        raise _fail()


def _validated_config(config: FairConfig) -> FairConfig:
    if type(config) is not FairConfig:
        raise _fail()
    current = vars(config)
    if (
        type(current) is not dict or any(type(key) is not str for key in current)
        or set(current) != set(_CONFIG_KEYS)
        or any(type(current[key]) is not str for key in _CONFIG_KEYS[:2])
        or any(type(current[key]) is not int for key in _CONFIG_KEYS[2:])
    ):
        raise _fail()
    FairConfig.__post_init__(config)
    return FairConfig(**current)


def _snapshot_fields(value: object, keys: set[str]) -> dict:
    if type(value) is not dict or not _native_tree(value) or set(value) != keys:
        raise _fail()
    return value


def _snapshot_dataset(sources: dict) -> FrozenDataset:
    """Construct native records only from the already pinned in-memory bytes.

    Raw SHA checks freeze every original value. Structural checks below keep
    the projection explicit; no legacy loader gets another opportunity to read.
    """
    manifest = sources["manifest_v1.json"]
    expected_manifest = {
        "dataset_id": "robotops-synthetic-v1", "version": "synthetic-1.0.0",
        "case_count": 50, "corpus_document_count": 15,
        "provenance": "SELF_AUTHORED_SYNTHETIC", "license_scope": "LOCAL_EVALUATION_ONLY",
        "publication_status": "NOT_AUTHORIZED", "cases_file": "cases_v1.json",
        "cases_sha256": _CASES_SHA256, "corpus_file": "corpus_v1.json",
        "corpus_sha256": _CORPUS_SHA256,
    }
    if not _same_native(manifest, expected_manifest):
        raise _fail()
    case_json = _snapshot_fields(sources["cases_v1.json"], {"version", "provenance", "license_scope", "cases"})
    corpus_json = _snapshot_fields(sources["corpus_v1.json"], {"version", "provenance", "license_scope", "documents"})
    for value in (case_json, corpus_json):
        if not _same_native({key: value[key] for key in ("version", "provenance", "license_scope")}, {
            "version": "synthetic-1.0.0", "provenance": "SELF_AUTHORED_SYNTHETIC",
            "license_scope": "LOCAL_EVALUATION_ONLY",
        }):
            raise _fail()
    if type(case_json["cases"]) is not list or len(case_json["cases"]) != 50:
        raise _fail()
    if type(corpus_json["documents"]) is not list or len(corpus_json["documents"]) != 15:
        raise _fail()
    docs = []
    for value in corpus_json["documents"]:
        doc = _snapshot_fields(value, {"id", "kind", "text"})
        if any(type(doc[key]) is not str for key in doc) or not 1 <= len(doc["text"]) <= 1000:
            raise _fail()
        if doc["kind"] not in {"SYNTH_MANUAL", "SYNTH_CASE", "SYNTH_LOG"}:
            raise _fail()
        docs.append(EvidenceDocument(evidence_id=doc["id"], kind=doc["kind"], text=doc["text"]))
    doc_ids = {doc.evidence_id for doc in docs}
    if len(doc_ids) != 15:
        raise _fail()
    cases = []
    scenarios = Counter()
    for number, value in enumerate(case_json["cases"], 1):
        case = _snapshot_fields(value, {"id", "scenario", "input", "gold"})
        if (
            any(type(case[key]) is not str for key in ("id", "scenario", "input"))
            or case["id"] != f"SYN-{number:03d}" or not 1 <= len(case["input"]) <= 300
        ):
            raise _fail()
        gold = _snapshot_fields(case["gold"], {"fault", "evidence_ids", "tools", "safety", "recovery"})
        if any(type(gold[key]) is not str for key in ("fault", "safety", "recovery")):
            raise _fail()
        for key, allowed in (("evidence_ids", doc_ids), ("tools", set(_TOOLS))):
            items = gold[key]
            if (type(items) is not list or any(type(item) is not str or item not in allowed for item in items)
                    or len(items) != len(set(items))):
                raise _fail()
        cases.append(EvaluationCase(
            case_id=case["id"], version="synthetic-1.0.0", scenario=case["scenario"], input=case["input"],
            gold=Gold(fault=gold["fault"], evidence_ids=tuple(gold["evidence_ids"]),
                      tools=tuple(gold["tools"]), safety=gold["safety"], recovery=gold["recovery"]),
            provenance="SELF_AUTHORED_SYNTHETIC", license_scope="LOCAL_EVALUATION_ONLY",
        ))
        scenarios[case["scenario"]] += 1
    if not _same_native(dict(scenarios), {
        "NORMAL": 20, "INFO_MISSING": 5, "HIGH_RISK": 5, "TOOL_FAILURE": 5,
        "RECOVERY": 5, "INJECTION": 5, "NO_ANSWER": 5,
    }):
        raise _fail()
    return FrozenDataset(
        dataset_id=manifest["dataset_id"], version=manifest["version"], cases_sha256=_CASES_SHA256,
        corpus_sha256=_CORPUS_SHA256, cases=tuple(cases), corpus=tuple(docs),
        scenario_counts=dict(scenarios), license_scope=manifest["license_scope"],
    )


def _prepare(config: FairConfig) -> tuple[FrozenDataset, FairConfig, dict]:
    try:
        safe_config = _validated_config(config)
        sources = {name: _decode_json(_source_bytes(name)) for name in _SOURCE_PINS}
        context = sources["context_plan_v2.json"]
        _validate_context(context)
        # The request and scoring sides both use this one verified snapshot.
        dataset = _snapshot_dataset(sources)
        return dataset, safe_config, context
    except Exception:
        raise _fail() from None


@dataclass(frozen=True)
class PublicContextRequest:
    input: str
    context_json: str
    corpus: tuple[PublicEvidence, ...]
    config: FairConfig


@dataclass(frozen=True)
class ContextPlannedRun:
    baseline: str
    case_id: str
    repetition: int
    request: PublicContextRequest
    dataset_id: str
    dataset_version: str
    cases_sha256: str
    corpus_sha256: str
    manifest_sha256: str
    parameter_formula_sha256: str
    context_version: str
    context_sha256: str
    context_fingerprint: str
    config_fingerprint: str
    public_input_sha256: str
    preparation_mode: str = "DATA_PREPARATION_ONLY"


def _extension_binding() -> dict[str, str]:
    return {"cases_sha256": _CASES_SHA256, "corpus_sha256": _CORPUS_SHA256,
            "manifest_sha256": _MANIFEST_SHA256, "parameter_formula_sha256": _PARAMETER_SHA256,
            "context_sha256": _CONTEXT_SHA256}


def _public_config(config: FairConfig) -> dict[str, str | int]:
    return {field.name: getattr(config, field.name) for field in fields(FairConfig)}


def _make_plan(dataset: FrozenDataset, config: FairConfig, context: dict) -> tuple[ContextPlannedRun, ...]:
    models = {row["case_id"]: row["device_model"] for row in context["cases"]}
    fingerprint = _digest(_extension_binding())
    result = []
    corpus = tuple(PublicEvidence(evidence_id=doc.evidence_id, text=doc.text) for doc in dataset.corpus)
    for repetition, case, baseline in (
        (repeat, item, name) for repeat in range(1, config.repetitions + 1)
        for item in dataset.cases for name in BASELINES
    ):
        context_json = _canonical({
            "device_model": models[case.case_id],
            "tool_catalog": context["tool_catalog"],
            "resource_catalog": context["resource_catalog"], "limits": context["limits"],
        })
        request = PublicContextRequest(input=case.input, context_json=context_json,
                                       corpus=() if baseline == "DIRECT_LLM" else corpus, config=config)
        public_digest = _digest({
            "input": request.input, "context_json": request.context_json,
            "corpus": [{"evidence_id": doc.evidence_id, "text": doc.text} for doc in request.corpus],
            "config": _public_config(request.config),
        })
        result.append(ContextPlannedRun(
            baseline=baseline, case_id=case.case_id, repetition=repetition,
            request=request, dataset_id=dataset.dataset_id, dataset_version=dataset.version,
            cases_sha256=_CASES_SHA256, corpus_sha256=_CORPUS_SHA256,
            manifest_sha256=_MANIFEST_SHA256, parameter_formula_sha256=_PARAMETER_SHA256,
            context_version=_CONTEXT_VERSION, context_sha256=_CONTEXT_SHA256,
            context_fingerprint=fingerprint, config_fingerprint=config.fingerprint,
            public_input_sha256=public_digest,
        ))
    return tuple(result)


def build_context_plan(config: FairConfig) -> tuple[ContextPlannedRun, ...]:
    """Prepare immutable public requests; this entry point never executes them."""
    dataset, safe_config, context = _prepare(config)
    return _make_plan(dataset, safe_config, context)


def build_context_readiness_report(config: FairConfig) -> dict[str, object]:
    """Return safe preparation metadata while every real metric stays pending."""
    dataset, safe_config, _ = _prepare(config)
    tool_counts = Counter(tool for case in dataset.cases for tool in case.gold.tools)
    applicable = sum(bool(case.gold.tools) for case in dataset.cases)
    expected_counts = dict(zip(_TOOLS, (30, 17, 0, 1, 17), strict=True))
    counts = {tool: tool_counts[tool] for tool in _TOOLS}
    if counts != expected_counts or applicable != 40 or sum(counts.values()) != 65:
        raise _fail()
    return {
        "report_kind": "LOCAL_DATA_CONTRACT_ONLY", "context_status": "CONTEXT_CONTRACT_READY",
        "admission_status": "BLOCKED_FOR_REAL_MEASUREMENT", "blockers": list(_BLOCKERS),
        "preparation_mode": "DATA_PREPARATION_ONLY", "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.version, "context_version": _CONTEXT_VERSION,
        **_extension_binding(), "context_fingerprint": _digest(_extension_binding()),
        "config_fingerprint": safe_config.fingerprint, "fair_config": _public_config(safe_config),
        "case_count": len(dataset.cases), "corpus_document_count": len(dataset.corpus),
        "repetitions": safe_config.repetitions,
        "planned_request_count": len(dataset.cases) * len(BASELINES) * safe_config.repetitions,
        "parameter_case_counts": {"applicable": applicable, "not_applicable": len(dataset.cases) - applicable},
        "expected_tool_selection_items": {"total": sum(counts.values()), "counts": counts,
                                          "status": "EXPECTED_SELECTION_ONLY"},
        "metrics": {metric: "PENDING_REAL_MEASUREMENT" for metric in _METRICS},
    }


class _SafeParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _fail()


def main() -> None:
    """Print one safe JSON summary, or a fixed error without input/path echoes."""
    try:
        parser = _SafeParser(prog="robotops-context-plan", add_help=False, allow_abbrev=False)
        parser.add_argument("--model-id", default="synthetic-context-plan")
        parser.add_argument("--temperature", default="0")
        for name, default in (("max-output-tokens", 512), ("timeout-ms", 30000),
                              ("max-tool-steps", 4), ("repetitions", 1)):
            parser.add_argument(f"--{name}", type=int, default=default)
        seen = set()
        for argument in sys.argv[1:]:
            if argument.startswith("--"):
                option = argument.split("=", 1)[0]
                if option in seen:
                    raise _fail()
                seen.add(option)
        arguments = parser.parse_args()
        report = build_context_readiness_report(FairConfig.from_mapping(vars(arguments)))
    except Exception:
        print("CONTEXT_PREPARATION_INVALID", file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps(report, ensure_ascii=True, sort_keys=True,
                     separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
