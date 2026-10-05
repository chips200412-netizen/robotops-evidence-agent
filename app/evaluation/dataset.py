"""Load and validate the frozen, self-authored offline evaluation dataset.

Only a deliberate public projection can be passed to model baselines. Gold,
scenario, safety, and recovery labels remain on the scoring side.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re


_VERSION = "synthetic-1.0.0"
_PROVENANCE = "SELF_AUTHORED_SYNTHETIC"
_LICENSE = "LOCAL_EVALUATION_ONLY"
_CASES_NAME = "cases_v1.json"
_CORPUS_NAME = "corpus_v1.json"
_SCENARIOS = {
    "NORMAL": 20, "INFO_MISSING": 5, "HIGH_RISK": 5,
    "TOOL_FAILURE": 5, "RECOVERY": 5, "INJECTION": 5,
    "NO_ANSWER": 5,
}
_FAULTS = frozenset({
    "POWER_UNDERVOLTAGE", "DRIVE_ENCODER_FEEDBACK",
    "NETWORK_PACKET_LOSS", "AUDIO_INPUT_MUTED", "SENSOR_CLOCK_SKEW",
    "UNKNOWN",
})
_READONLY_TOOLS = frozenset({
    "search_manual", "search_fault_cases", "query_device_information",
    "query_simulated_telemetry", "analyze_synthetic_ros_log",
})
_SAFETY = frozenset({
    "ALLOW_READONLY", "ASK_MORE", "REQUIRE_APPROVAL",
    "REFUSE_INJECTION", "UNKNOWN",
})
_RECOVERY = frozenset({"NONE", "DEGRADE", "RESUME"})
_DOC_KINDS = frozenset({"SYNTH_MANUAL", "SYNTH_CASE", "SYNTH_LOG"})
_FORBIDDEN = re.compile(
    r"tonybot|wxid_|sk-[a-z0-9]{8,}|\.env|[a-z]:[\\/]|\b1[3-9]\d{9}\b|"
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|"
    r"\bSN[-_: ]\w{6,}\b|"
    r"\b(?:api[_-]?key|access[_-]?token|secret[_-]?key|password)\s*[:=]\s*\S+",
    re.I,
)


def _fail() -> ValueError:
    return ValueError("invalid frozen evaluation dataset")


def _obj(value: object, keys: set[str]) -> dict:
    if type(value) is not dict or set(value) != keys:
        raise _fail()
    return value


def _safe_text(value: object, *, maximum: int) -> str:
    if (
        type(value) is not str or not 1 <= len(value) <= maximum
        or not value.strip() or _FORBIDDEN.search(value)
    ):
        raise _fail()
    return value


def _load_json_bytes(root: Path, name: str, expected_sha256: str) -> dict:
    if type(expected_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise _fail()
    raw = (root / name).read_bytes()
    if sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("frozen dataset digest mismatch")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _fail() from exc
    if type(value) is not dict:
        raise _fail()
    return value


@dataclass(frozen=True)
class Gold:
    fault: str
    evidence_ids: tuple[str, ...]
    tools: tuple[str, ...]
    safety: str
    recovery: str


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    version: str
    scenario: str
    input: str
    gold: Gold
    provenance: str
    license_scope: str


@dataclass(frozen=True)
class EvidenceDocument:
    evidence_id: str
    kind: str
    text: str


@dataclass(frozen=True)
class FrozenDataset:
    dataset_id: str
    version: str
    cases_sha256: str
    corpus_sha256: str
    cases: tuple[EvaluationCase, ...]
    corpus: tuple[EvidenceDocument, ...]
    scenario_counts: dict[str, int]
    license_scope: str

    def public_case_inputs(self) -> tuple[dict[str, str], ...]:
        """No gold, safety, scenario, license, or provenance in model input."""
        return tuple({"case_id": case.case_id, "input": case.input}
                     for case in self.cases)

    def public_corpus(self) -> tuple[dict[str, str], ...]:
        """Only self-authored read-only evidence, with stable reference IDs."""
        return tuple({"evidence_id": doc.evidence_id, "text": doc.text}
                     for doc in self.corpus)


def load_frozen_dataset(root: Path) -> FrozenDataset:
    """Fail closed on byte drift, invalid gold links, or unsafe source metadata."""
    if not isinstance(root, Path):
        raise _fail()
    try:
        manifest_value = json.loads((root / "manifest_v1.json").read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _fail() from exc
    manifest = _obj(manifest_value, {
        "dataset_id", "version", "case_count", "corpus_document_count",
        "provenance", "license_scope", "publication_status",
        "cases_file", "cases_sha256", "corpus_file", "corpus_sha256",
    })
    if (
        manifest["dataset_id"] != "robotops-synthetic-v1"
        or manifest["version"] != _VERSION
        or type(manifest["case_count"]) is not int or manifest["case_count"] != 50
        or type(manifest["corpus_document_count"]) is not int
        or manifest["corpus_document_count"] != 15
        or manifest["provenance"] != _PROVENANCE
        or manifest["license_scope"] != _LICENSE
        or manifest["publication_status"] != "NOT_AUTHORIZED"
        or manifest["cases_file"] != _CASES_NAME
        or manifest["corpus_file"] != _CORPUS_NAME
    ):
        raise _fail()
    case_json = _load_json_bytes(root, _CASES_NAME, manifest["cases_sha256"])
    corpus_json = _load_json_bytes(root, _CORPUS_NAME, manifest["corpus_sha256"])
    _obj(case_json, {"version", "provenance", "license_scope", "cases"})
    _obj(corpus_json, {"version", "provenance", "license_scope", "documents"})
    for value in (case_json, corpus_json):
        if (
            value["version"] != _VERSION
            or value["provenance"] != _PROVENANCE
            or value["license_scope"] != _LICENSE
        ):
            raise _fail()
    if type(corpus_json["documents"]) is not list or len(corpus_json["documents"]) != 15:
        raise _fail()
    docs: list[EvidenceDocument] = []
    doc_ids: set[str] = set()
    for item in corpus_json["documents"]:
        item = _obj(item, {"id", "kind", "text"})
        evidence_id = item["id"]
        if (
            type(evidence_id) is not str
            or re.fullmatch(r"(?:DOC|CASE|LOG)-[A-Z]+-01", evidence_id) is None
            or evidence_id in doc_ids
            or type(item["kind"]) is not str or item["kind"] not in _DOC_KINDS
            or (evidence_id.startswith("DOC-") and item["kind"] != "SYNTH_MANUAL")
            or (evidence_id.startswith("CASE-") and item["kind"] != "SYNTH_CASE")
            or (evidence_id.startswith("LOG-") and item["kind"] != "SYNTH_LOG")
        ):
            raise _fail()
        doc_ids.add(evidence_id)
        docs.append(EvidenceDocument(
            evidence_id=evidence_id, kind=item["kind"],
            text=_safe_text(item["text"], maximum=1000),
        ))
    if type(case_json["cases"]) is not list or len(case_json["cases"]) != 50:
        raise _fail()
    cases: list[EvaluationCase] = []
    scenarios: Counter[str] = Counter()
    normal_faults: Counter[str] = Counter()
    for index, item in enumerate(case_json["cases"], 1):
        item = _obj(item, {"id", "scenario", "input", "gold"})
        case_id = f"SYN-{index:03d}"
        scenario = item["scenario"]
        if item["id"] != case_id or type(scenario) is not str or scenario not in _SCENARIOS:
            raise _fail()
        gold = _obj(item["gold"], {
            "fault", "evidence_ids", "tools", "safety", "recovery",
        })
        if (
            type(gold["fault"]) is not str or gold["fault"] not in _FAULTS
            or type(gold["evidence_ids"]) is not list
            or any(type(eid) is not str or eid not in doc_ids
                   for eid in gold["evidence_ids"])
            or len(gold["evidence_ids"]) != len(set(gold["evidence_ids"]))
            or type(gold["tools"]) is not list
            or any(type(tool) is not str or tool not in _READONLY_TOOLS
                   for tool in gold["tools"])
            or len(gold["tools"]) != len(set(gold["tools"]))
            or type(gold["safety"]) is not str or gold["safety"] not in _SAFETY
            or type(gold["recovery"]) is not str or gold["recovery"] not in _RECOVERY
        ):
            raise _fail()
        if gold["fault"] == "UNKNOWN" and (gold["evidence_ids"] or gold["tools"]):
            raise _fail()
        expected = {
            "NORMAL": ("ALLOW_READONLY", "NONE"),
            "INFO_MISSING": ("ASK_MORE", "NONE"),
            "HIGH_RISK": ("REQUIRE_APPROVAL", "NONE"),
            "TOOL_FAILURE": ("ALLOW_READONLY", "DEGRADE"),
            "RECOVERY": ("ALLOW_READONLY", "RESUME"),
            "INJECTION": ("REFUSE_INJECTION", "NONE"),
            "NO_ANSWER": ("UNKNOWN", "NONE"),
        }[scenario]
        if (gold["safety"], gold["recovery"]) != expected:
            raise _fail()
        if scenario in {"INFO_MISSING", "NO_ANSWER"}:
            if gold["fault"] != "UNKNOWN":
                raise _fail()
        elif gold["fault"] == "UNKNOWN" or not gold["evidence_ids"]:
            raise _fail()
        if scenario == "TOOL_FAILURE" and len(gold["tools"]) < 2:
            raise _fail()
        if scenario == "NORMAL":
            normal_faults[gold["fault"]] += 1
        scenarios[scenario] += 1
        cases.append(EvaluationCase(
            case_id=case_id, version=_VERSION, scenario=scenario,
            input=_safe_text(item["input"], maximum=300),
            gold=Gold(
                fault=gold["fault"], evidence_ids=tuple(gold["evidence_ids"]),
                tools=tuple(gold["tools"]), safety=gold["safety"],
                recovery=gold["recovery"],
            ),
            provenance=_PROVENANCE, license_scope=_LICENSE,
        ))
    if scenarios != _SCENARIOS or any(
        normal_faults[fault] < 4 for fault in _FAULTS if fault != "UNKNOWN"
    ):
        raise _fail()
    return FrozenDataset(
        dataset_id=manifest["dataset_id"], version=_VERSION,
        cases_sha256=manifest["cases_sha256"],
        corpus_sha256=manifest["corpus_sha256"], cases=tuple(cases),
        corpus=tuple(docs), scenario_counts=dict(scenarios),
        license_scope=_LICENSE,
    )


def main() -> None:
    """Offline integrity check; prints metadata and coverage, never gold rows."""
    root = Path(__file__).resolve().parents[2] / "eval"
    dataset = load_frozen_dataset(root)
    print(json.dumps({
        "dataset_id": dataset.dataset_id, "version": dataset.version,
        "case_count": len(dataset.cases), "corpus_document_count": len(dataset.corpus),
        "scenario_counts": dataset.scenario_counts,
        "cases_sha256": dataset.cases_sha256,
        "corpus_sha256": dataset.corpus_sha256,
        "license_scope": dataset.license_scope,
        "publication_status": "NOT_AUTHORIZED",
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
