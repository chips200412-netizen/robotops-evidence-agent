"""MOD-023 本地合成遥测与 ROS 日志读取器。

本模块只读取项目内固定 manifest 所列的合成 JSONL。它不连接真实机器人、
ROS、网络、数据库或外部进程，也不声称提供通用操作系统沙箱。
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import secrets
import stat
from types import MappingProxyType
from typing import Any, BinaryIO
import unicodedata


SIMULATED_TELEMETRY_TOOL = "query_simulated_telemetry"
SYNTHETIC_ROS_LOG_TOOL = "analyze_synthetic_ros_log"
SUPPORTED_TOOL_NAMES = frozenset(
    {SIMULATED_TELEMETRY_TOOL, SYNTHETIC_ROS_LOG_TOOL}
)

MAX_TELEMETRY_POINTS = 120
MAX_ROS_LOG_ENTRIES = 40
MAX_EVIDENCE_BYTES = 32_768
MAX_TIME_WINDOW_MS = 600_000
MAX_JSONL_LINE_BYTES = 4_096
MAX_RESULT_BYTES = 64 * 1_024
MAX_LOG_MESSAGE_CHARS = 1_000
RESULT_SCHEMA_VERSION = 1
RESULT_SOURCE = "SYNTHETIC_LOCAL"
RESULT_TRUST_BOUNDARY = "UNTRUSTED_DATA"

SUPPORTED_DEVICE_MODELS = (
    "EduBot A1",
    "EduBot A2 Pro",
    "EduCar X1",
    "Humanoid H1",
    "夸父 4 Pro",
    "Tonybot",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = PROJECT_ROOT / "data" / "synthetic_robotops"
MANIFEST_NAME = "manifest.json"
MANIFEST_MAX_BYTES = 32_768

TOOL_ARGUMENT_INVALID = "TOOL_ARGUMENT_INVALID"
TOOL_SCOPE_INVALID = "TOOL_SCOPE_INVALID"
TOOL_TIME_WINDOW_INVALID = "TOOL_TIME_WINDOW_INVALID"
TOOL_PATH_INVALID = "TOOL_PATH_INVALID"
TOOL_INPUT_TOO_LARGE = "TOOL_INPUT_TOO_LARGE"
TOOL_INPUT_ENCODING_INVALID = "TOOL_INPUT_ENCODING_INVALID"
TOOL_INPUT_FORMAT_INVALID = "TOOL_INPUT_FORMAT_INVALID"
TOOL_SOURCE_CHANGED = "TOOL_SOURCE_CHANGED"
TOOL_OUTPUT_INVALID = "TOOL_OUTPUT_INVALID"
TOOL_EXECUTION_ERROR = "TOOL_EXECUTION_ERROR"
TOOL_UNKNOWN = "TOOL_UNKNOWN"

STABLE_ERROR_CODES = frozenset(
    {
        TOOL_ARGUMENT_INVALID,
        TOOL_SCOPE_INVALID,
        TOOL_TIME_WINDOW_INVALID,
        TOOL_PATH_INVALID,
        TOOL_INPUT_TOO_LARGE,
        TOOL_INPUT_ENCODING_INVALID,
        TOOL_INPUT_FORMAT_INVALID,
        TOOL_SOURCE_CHANGED,
        TOOL_OUTPUT_INVALID,
        TOOL_EXECUTION_ERROR,
        TOOL_UNKNOWN,
    }
)

_RESOURCE_ID_RE = re.compile(r"^[a-z][a-z0-9-]{2,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PATH_COMPONENT_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_NODE_RE = re.compile(r"^/[a-z0-9_]+(?:/[a-z0-9_]+)*$")
_RESERVED_WINDOWS_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
)
_COMPRESSED_MAGICS = (
    b"PK\x03\x04",
    b"PK\x05\x06",
    b"PK\x07\x08",
    b"\x1f\x8b",
    b"BZh",
    b"\xfd7zXZ\x00",
)
_METRIC_UNITS = MappingProxyType(
    {
        "battery_voltage": "V",
        "controller_temperature": "°C",
        "imu_vibration": "m/s²",
        "joint_error": "deg",
        "motor_current": "A",
        "network_latency": "ms",
    }
)
_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARN", "ERROR", "FATAL"})
_TELEMETRY_QUALITIES = frozenset({"GOOD", "DEGRADED", "BAD"})
_SENSITIVE_PATTERNS = (
    re.compile(
        r"-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----.*?"
        r"(?:-----END(?: [A-Z0-9]+)? PRIVATE KEY-----|$)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(?i)\b(?:proxy-)?authorization\s*:\s*"
        r"(?:basic|bearer|token)\s+[a-z0-9+/._~=-]{4,}"
    ),
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|password|passwd|secret)"
        r"\s*[:=]\s*[^\s,;]+"
    ),
    re.compile(
        r"(?i)\b(?:[a-z][a-z0-9_]*_)?(?:api_key|access_token|password|passwd|"
        r"secret|token)\s*[:=]\s*[^\s,;]+"
    ),
    re.compile(
        r"(?i)\baws_(?:secret_access_key|access_key_id|session_token)"
        r"\s*[:=]\s*[^\s,;]+"
    ),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\bgithub_pat_[a-z0-9_]{20,}"),
    re.compile(r"(?i)\bsk-[a-z0-9_-]{8,}"),
    re.compile(r"(?i)\bgh[pousr]_[a-z0-9]{20,}"),
    re.compile(r"(?i)\bxox[baprs]-[a-z0-9-]{20,}"),
    re.compile(r"(?i)\bglpat-[a-z0-9_-]{20,}"),
    re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s/@:]+:[^\s/@]+@"),
    re.compile(r"(?i)\\\\[^\\\s,;]+\\[^\s,;]+"),
    re.compile(r"(?i)\b[a-z]:\\(?:[^\\\s]+\\)*[^\\\s]*"),
    re.compile(
        r"(?i)(?<![a-z0-9])/(?:app|bin|boot|data|dev|etc|home|lib|lib64|"
        r"media|mnt|opt|proc|root|run|sbin|srv|sys|tmp|usr|users|var|workspace)"
        r"(?:/[^\s,;]+)?"
    ),
    re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b"),
    re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"),
    re.compile(r"(?i)(?<![0-9a-f])(?:[0-9a-f]{2}:){5}[0-9a-f]{2}(?![0-9a-f])"),
)

_EXPECTED_LAYOUT = MappingProxyType(
    {
        "EduBot A1": MappingProxyType(
            {
                "telemetry": (
                    "telemetry-edubot-a1-v1",
                    "telemetry/edubot_a1.jsonl",
                ),
                "ros_log": (
                    "roslog-edubot-a1-v1",
                    "ros_logs/edubot_a1.jsonl",
                ),
            }
        ),
        "EduBot A2 Pro": MappingProxyType(
            {
                "telemetry": (
                    "telemetry-edubot-a2-pro-v1",
                    "telemetry/edubot_a2_pro.jsonl",
                ),
                "ros_log": (
                    "roslog-edubot-a2-pro-v1",
                    "ros_logs/edubot_a2_pro.jsonl",
                ),
            }
        ),
        "EduCar X1": MappingProxyType(
            {
                "telemetry": (
                    "telemetry-educar-x1-v1",
                    "telemetry/educar_x1.jsonl",
                ),
                "ros_log": (
                    "roslog-educar-x1-v1",
                    "ros_logs/educar_x1.jsonl",
                ),
            }
        ),
        "Humanoid H1": MappingProxyType(
            {
                "telemetry": (
                    "telemetry-humanoid-h1-v1",
                    "telemetry/humanoid_h1.jsonl",
                ),
                "ros_log": (
                    "roslog-humanoid-h1-v1",
                    "ros_logs/humanoid_h1.jsonl",
                ),
            }
        ),
        "夸父 4 Pro": MappingProxyType(
            {
                "telemetry": (
                    "telemetry-kuafu-4-pro-v1",
                    "telemetry/kuafu_4_pro.jsonl",
                ),
                "ros_log": (
                    "roslog-kuafu-4-pro-v1",
                    "ros_logs/kuafu_4_pro.jsonl",
                ),
            }
        ),
        "Tonybot": MappingProxyType(
            {
                "telemetry": (
                    "telemetry-tonybot-v1",
                    "telemetry/tonybot.jsonl",
                ),
                "ros_log": (
                    "roslog-tonybot-v1",
                    "ros_logs/tonybot.jsonl",
                ),
            }
        ),
    }
)


class EvidenceToolError(ValueError):
    """Only exposes a stable code; paths and source text never enter the error."""

    def __init__(self, code: str) -> None:
        self.code = code if code in STABLE_ERROR_CODES else TOOL_EXECUTION_ERROR
        super().__init__(self.code)


@dataclass(frozen=True)
class EvidenceResource:
    kind: str
    resource_id: str
    device_model: str
    relative_path: str
    start_ms: int
    end_ms: int
    size_bytes: int
    sha256: str


def _raise(code: str) -> None:
    raise EvidenceToolError(code) from None


def _strict_int(value: object, *, code: str) -> int:
    if type(value) is not int or not 0 <= value <= (2**63 - 1):
        _raise(code)
    return value


def _bounded_safe_text(value: object, *, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        _raise(TOOL_INPUT_FORMAT_INVALID)
    for character in value:
        category = unicodedata.category(character)
        if category in {"Cc", "Cf", "Cs"}:
            _raise(TOOL_INPUT_ENCODING_INVALID)
    return value


def _redact_sensitive(value: str) -> str:
    redacted = value
    for pattern in _SENSITIVE_PATTERNS:
        redacted = pattern.sub("[REDACTED_SENSITIVE]", redacted)
    return redacted


def _canonical_json_bytes(value: object, *, error_code: str) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        _raise(error_code)
    if len(encoded) > MAX_RESULT_BYTES:
        _raise(TOOL_OUTPUT_INVALID)
    return encoded


def _validate_relative_path(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 240
        or "\\" in value
        or ":" in value
        or value.startswith("/")
        or value.endswith((".", " "))
        or "//" in value
    ):
        _raise(TOOL_PATH_INVALID)
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        _raise(TOOL_PATH_INVALID)
    windows = PureWindowsPath(value)
    posix = PurePosixPath(value)
    if windows.is_absolute() or windows.drive or windows.root or posix.is_absolute():
        _raise(TOOL_PATH_INVALID)
    parts = posix.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        _raise(TOOL_PATH_INVALID)
    for part in parts:
        if (
            part.endswith((".", " "))
            or _PATH_COMPONENT_RE.fullmatch(part) is None
            or part.split(".", 1)[0].upper() in _RESERVED_WINDOWS_NAMES
        ):
            _raise(TOOL_PATH_INVALID)
    if posix.suffix != ".jsonl" and value != MANIFEST_NAME:
        _raise(TOOL_PATH_INVALID)
    return value


def _is_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
        attributes = int(getattr(info, "st_file_attributes", 0))
        reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
        return bool(
            path.is_symlink()
            or (hasattr(path, "is_junction") and path.is_junction())
            or attributes & reparse_flag
        )
    except OSError:
        _raise(TOOL_PATH_INVALID)


def _evidence_root() -> Path:
    try:
        project = PROJECT_ROOT.resolve(strict=True)
    except OSError:
        _raise(TOOL_PATH_INVALID)
    if not project.is_dir():
        _raise(TOOL_PATH_INVALID)
    configured = EVIDENCE_ROOT.absolute()
    try:
        lexical_parts = configured.relative_to(project).parts
    except ValueError:
        _raise(TOOL_PATH_INVALID)
    current = project
    for part in lexical_parts:
        current = current / part
        if _is_reparse(current):
            _raise(TOOL_PATH_INVALID)
    try:
        root = configured.resolve(strict=True)
        root.relative_to(project)
    except (OSError, ValueError):
        _raise(TOOL_PATH_INVALID)
    if not root.is_dir():
        _raise(TOOL_PATH_INVALID)
    return root


def _resolve_resource(relative_path: object) -> tuple[Path, Path]:
    safe_relative = _validate_relative_path(relative_path)
    root = _evidence_root()
    candidate = root.joinpath(*PurePosixPath(safe_relative).parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError):
        _raise(TOOL_PATH_INVALID)
    current = root
    if _is_reparse(current):
        _raise(TOOL_PATH_INVALID)
    for part in PurePosixPath(safe_relative).parts:
        current = current / part
        if _is_reparse(current):
            _raise(TOOL_PATH_INVALID)
    try:
        if not resolved.is_file() or not stat.S_ISREG(resolved.stat().st_mode):
            _raise(TOOL_PATH_INVALID)
    except OSError:
        _raise(TOOL_PATH_INVALID)
    return root, resolved


def _normalize_windows_final_path(value: str) -> str:
    if value.startswith("\\\\?\\UNC\\"):
        candidate = "\\\\" + value[8:]
        parts = PureWindowsPath(candidate).parts
        if len(parts) < 2 or not PureWindowsPath(candidate).is_absolute():
            _raise(TOOL_PATH_INVALID)
        return candidate
    if value.startswith("\\\\?\\"):
        candidate = value[4:]
        if re.fullmatch(r"[A-Za-z]:\\.*", candidate) is None:
            _raise(TOOL_PATH_INVALID)
        return candidate
    candidate_path = PureWindowsPath(value)
    if not candidate_path.is_absolute() or not candidate_path.drive:
        _raise(TOOL_PATH_INVALID)
    if value.startswith("\\\\.\\") or value.startswith("\\??\\"):
        _raise(TOOL_PATH_INVALID)
    return value


def _contained_path_text(candidate: str, root: Path) -> bool:
    normalized_candidate = os.path.normcase(os.path.abspath(candidate))
    normalized_root = os.path.normcase(os.path.abspath(str(root)))
    try:
        return os.path.commonpath((normalized_candidate, normalized_root)) == normalized_root
    except ValueError:
        return False


def _open_windows_binary(path: Path, root: Path) -> BinaryIO:
    """Open one Windows file by handle and verify its resolved handle path.

    This closes the named-path race for this local read. It is not a process or
    container sandbox and does not defend against an administrator replacing ACLs.
    """

    import msvcrt

    class _FileTime(ctypes.Structure):
        _fields_ = (
            ("low", ctypes.c_uint32),
            ("high", ctypes.c_uint32),
        )

    class _ByHandleFileInformation(ctypes.Structure):
        _fields_ = (
            ("file_attributes", ctypes.c_uint32),
            ("creation_time", _FileTime),
            ("last_access_time", _FileTime),
            ("last_write_time", _FileTime),
            ("volume_serial_number", ctypes.c_uint32),
            ("file_size_high", ctypes.c_uint32),
            ("file_size_low", ctypes.c_uint32),
            ("number_of_links", ctypes.c_uint32),
            ("file_index_high", ctypes.c_uint32),
            ("file_index_low", ctypes.c_uint32),
        )

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    )
    create_file.restype = ctypes.c_void_p
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (ctypes.c_void_p,)
    close_handle.restype = ctypes.c_int
    get_final_path = kernel32.GetFinalPathNameByHandleW
    get_final_path.argtypes = (
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
    )
    get_final_path.restype = ctypes.c_uint32
    get_file_information = kernel32.GetFileInformationByHandle
    get_file_information.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(_ByHandleFileInformation),
    )
    get_file_information.restype = ctypes.c_int

    generic_read = 0x80000000
    file_share_read = 0x00000001
    open_existing = 3
    file_attribute_normal = 0x00000080
    file_flag_open_reparse_point = 0x00200000
    file_flag_sequential_scan = 0x08000000
    invalid_handle = ctypes.c_void_p(-1).value
    handle = create_file(
        str(path),
        generic_read,
        file_share_read,
        None,
        open_existing,
        file_attribute_normal
        | file_flag_open_reparse_point
        | file_flag_sequential_scan,
        None,
    )
    if handle in {None, invalid_handle}:
        _raise(TOOL_PATH_INVALID)
    transferred = False
    try:
        information = _ByHandleFileInformation()
        if not get_file_information(handle, ctypes.byref(information)):
            _raise(TOOL_PATH_INVALID)
        if information.file_attributes & (0x00000400 | 0x00000010):
            _raise(TOOL_PATH_INVALID)
        size = 512
        while size <= 32_768:
            buffer = ctypes.create_unicode_buffer(size)
            length = get_final_path(handle, buffer, size, 0)
            if length == 0:
                _raise(TOOL_PATH_INVALID)
            if length < size:
                final_path = _normalize_windows_final_path(buffer.value)
                if not _contained_path_text(final_path, root):
                    _raise(TOOL_PATH_INVALID)
                break
            size = int(length) + 1
        else:
            _raise(TOOL_PATH_INVALID)

        descriptor = msvcrt.open_osfhandle(
            int(handle), os.O_RDONLY | getattr(os, "O_BINARY", 0)
        )
        try:
            stream = os.fdopen(descriptor, "rb", closefd=True)
        except Exception:
            try:
                os.close(descriptor)
            finally:
                transferred = True
            raise
        transferred = True
        return stream
    except EvidenceToolError:
        raise
    except (OSError, ValueError):
        _raise(TOOL_PATH_INVALID)
    finally:
        if not transferred:
            close_handle(handle)


@contextmanager
def _open_verified_binary(path: Path, root: Path) -> Iterator[BinaryIO]:
    stream: BinaryIO | None = None
    try:
        if os.name == "nt":
            stream = _open_windows_binary(path, root)
        else:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(path, flags)
            stream = os.fdopen(descriptor, "rb", closefd=True)
        yield stream
    except EvidenceToolError:
        raise
    except OSError:
        _raise(TOOL_PATH_INVALID)
    finally:
        if stream is not None:
            stream.close()


def _read_bounded_resource(
    relative_path: object,
    *,
    maximum_bytes: int,
    expected_size_bytes: int | None,
    expected_sha256: str | None,
) -> tuple[bytes, str]:
    root, path = _resolve_resource(relative_path)
    with _open_verified_binary(path, root) as stream:
        try:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                _raise(TOOL_PATH_INVALID)
            if before.st_size < 0 or before.st_size > maximum_bytes:
                _raise(TOOL_INPUT_TOO_LARGE)
            if (
                expected_size_bytes is not None
                and before.st_size != expected_size_bytes
            ):
                _raise(TOOL_SOURCE_CHANGED)
            payload = stream.read(maximum_bytes + 1)
            after = os.fstat(stream.fileno())
        except EvidenceToolError:
            raise
        except OSError:
            _raise(TOOL_SOURCE_CHANGED)
    if len(payload) > maximum_bytes:
        _raise(TOOL_INPUT_TOO_LARGE)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity or len(payload) != before.st_size:
        _raise(TOOL_SOURCE_CHANGED)
    if expected_size_bytes is not None and len(payload) != expected_size_bytes:
        _raise(TOOL_SOURCE_CHANGED)
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and not secrets.compare_digest(
        digest, expected_sha256
    ):
        _raise(TOOL_SOURCE_CHANGED)
    return payload, digest


def _decode_jsonl(payload: bytes) -> list[dict[str, Any]]:
    if any(payload.startswith(magic) for magic in _COMPRESSED_MAGICS):
        _raise(TOOL_INPUT_FORMAT_INVALID)
    if b"\r" in payload.replace(b"\r\n", b""):
        _raise(TOOL_INPUT_FORMAT_INVALID)
    raw_lines = payload.split(b"\n")
    if raw_lines and raw_lines[-1] == b"":
        raw_lines.pop()
    if not raw_lines or any(not line for line in raw_lines):
        _raise(TOOL_INPUT_FORMAT_INVALID)
    result: list[dict[str, Any]] = []

    def reject_constant(_value: str) -> None:
        raise ValueError("non-finite JSON constant")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        item: dict[str, Any] = {}
        for key, value in pairs:
            if key in item:
                raise ValueError("duplicate JSON key")
            item[key] = value
        return item

    for raw_line in raw_lines:
        if raw_line.endswith(b"\r"):
            raw_line = raw_line[:-1]
        if not raw_line or len(raw_line) > MAX_JSONL_LINE_BYTES:
            _raise(TOOL_INPUT_TOO_LARGE)
        try:
            line = raw_line.decode("utf-8", errors="strict")
            parsed = json.loads(
                line,
                object_pairs_hook=unique_object,
                parse_constant=reject_constant,
            )
        except (UnicodeDecodeError, UnicodeError):
            _raise(TOOL_INPUT_ENCODING_INVALID)
        except (json.JSONDecodeError, ValueError):
            _raise(TOOL_INPUT_FORMAT_INVALID)
        if not isinstance(parsed, dict):
            _raise(TOOL_INPUT_FORMAT_INVALID)
        result.append(parsed)
    return result


def _decode_json_object(payload: bytes) -> dict[str, Any]:
    if any(payload.startswith(magic) for magic in _COMPRESSED_MAGICS):
        _raise(TOOL_INPUT_FORMAT_INVALID)

    def reject_constant(_value: str) -> None:
        raise ValueError("non-finite JSON constant")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        item: dict[str, Any] = {}
        for key, value in pairs:
            if key in item:
                raise ValueError("duplicate JSON key")
            item[key] = value
        return item

    try:
        text = payload.decode("utf-8", errors="strict")
        parsed = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, UnicodeError):
        _raise(TOOL_INPUT_ENCODING_INVALID)
    except (json.JSONDecodeError, ValueError):
        _raise(TOOL_INPUT_FORMAT_INVALID)
    if not isinstance(parsed, dict):
        _raise(TOOL_INPUT_FORMAT_INVALID)
    return parsed


def _resource_from_manifest(
    value: object,
    *,
    expected_kind: str,
    expected_id: str,
    expected_device_model: str,
    expected_path: str,
) -> EvidenceResource:
    if not isinstance(value, dict) or set(value) != {
        "resource_id",
        "kind",
        "device_model",
        "path",
        "start_ms",
        "end_ms",
        "size_bytes",
        "sha256",
    }:
        _raise(TOOL_INPUT_FORMAT_INVALID)
    if value.get("kind") != expected_kind:
        _raise(TOOL_INPUT_FORMAT_INVALID)
    if value.get("device_model") != expected_device_model:
        _raise(TOOL_SCOPE_INVALID)

    resource_id = value.get("resource_id")
    relative_path = value.get("path")
    digest = value.get("sha256")
    if (
        not isinstance(resource_id, str)
        or _RESOURCE_ID_RE.fullmatch(resource_id) is None
        or resource_id != expected_id
    ):
        _raise(TOOL_SCOPE_INVALID)
    if relative_path != expected_path:
        _raise(TOOL_PATH_INVALID)
    safe_path = _validate_relative_path(relative_path)
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        _raise(TOOL_INPUT_FORMAT_INVALID)

    start_ms = _strict_int(value.get("start_ms"), code=TOOL_TIME_WINDOW_INVALID)
    end_ms = _strict_int(value.get("end_ms"), code=TOOL_TIME_WINDOW_INVALID)
    if start_ms >= end_ms or end_ms - start_ms > MAX_TIME_WINDOW_MS:
        _raise(TOOL_TIME_WINDOW_INVALID)
    size_bytes = _strict_int(
        value.get("size_bytes"), code=TOOL_INPUT_FORMAT_INVALID
    )
    if size_bytes <= 0 or size_bytes > MAX_EVIDENCE_BYTES:
        _raise(TOOL_INPUT_TOO_LARGE)

    return EvidenceResource(
        kind=expected_kind,
        resource_id=resource_id,
        device_model=expected_device_model,
        relative_path=safe_path,
        start_ms=start_ms,
        end_ms=end_ms,
        size_bytes=size_bytes,
        sha256=digest,
    )


def _load_manifest() -> dict[tuple[str, str], EvidenceResource]:
    payload, _digest = _read_bounded_resource(
        MANIFEST_NAME,
        maximum_bytes=MANIFEST_MAX_BYTES,
        expected_size_bytes=None,
        expected_sha256=None,
    )
    manifest = _decode_json_object(payload)
    if (
        set(manifest) != {"schema_version", "resources"}
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or not isinstance(manifest.get("resources"), list)
        or len(manifest["resources"]) != len(SUPPORTED_DEVICE_MODELS) * 2
    ):
        _raise(TOOL_INPUT_FORMAT_INVALID)

    resources: dict[tuple[str, str], EvidenceResource] = {}
    resource_ids: set[str] = set()
    for item in manifest["resources"]:
        if not isinstance(item, dict):
            _raise(TOOL_INPUT_FORMAT_INVALID)
        device_model = item.get("device_model")
        kind = item.get("kind")
        if device_model not in _EXPECTED_LAYOUT or kind not in {"telemetry", "ros_log"}:
            _raise(TOOL_SCOPE_INVALID)
        key = (device_model, kind)
        if key in resources:
            _raise(TOOL_SCOPE_INVALID)
        layout = _EXPECTED_LAYOUT[device_model][kind]
        resource = _resource_from_manifest(
            item,
            expected_kind=kind,
            expected_id=layout[0],
            expected_device_model=device_model,
            expected_path=layout[1],
        )
        if resource.resource_id in resource_ids:
            _raise(TOOL_SCOPE_INVALID)
        resource_ids.add(resource.resource_id)
        resources[key] = resource

    expected_keys = {
        (device_model, kind)
        for device_model in SUPPORTED_DEVICE_MODELS
        for kind in ("telemetry", "ros_log")
    }
    if set(resources) != expected_keys:
        _raise(TOOL_SCOPE_INVALID)
    return resources


def freeze_resource_metadata() -> list[dict[str, Any]]:
    """Read the manifest once and detach its complete, path-free metadata."""
    resources = _load_manifest()
    return validate_resource_metadata([
        {
            "kind": resource.kind,
            "resource_id": resource.resource_id,
            "device_model": resource.device_model,
            "start_ms": resource.start_ms,
            "end_ms": resource.end_ms,
            "size_bytes": resource.size_bytes,
            "sha256": resource.sha256,
        }
        for _key, resource in sorted(resources.items())
    ])


def validate_resource_metadata(value: object) -> list[dict[str, Any]]:
    """Pure validation of all twelve exact, sorted, path-free resource records.

    The fixed shallow schema bounds traversal before copying. No caller-owned
    container or scalar subtype is accepted, and metadata never grants file access.
    """
    expected_pairs = sorted(
        (model, kind) for model, kinds in _EXPECTED_LAYOUT.items() for kind in kinds
    )
    if type(value) is not list or len(value) != len(expected_pairs):
        _raise(TOOL_INPUT_FORMAT_INVALID)
    keys = {"kind", "resource_id", "device_model", "start_ms", "end_ms", "size_bytes", "sha256"}
    validated = []
    for item, (model, kind) in zip(value, expected_pairs):
        if (
            type(item) is not dict
            or len(item) != len(keys)
            or any(type(key) is not str for key in item)
            or set(item) != keys
            or any(type(item[key]) is not str for key in ("kind", "resource_id", "device_model", "sha256"))
        ):
            _raise(TOOL_INPUT_FORMAT_INVALID)
        resource_id, relative_path = _EXPECTED_LAYOUT[model][kind]
        resource = _resource_from_manifest(
            {**item, "path": relative_path},
            expected_kind=kind,
            expected_id=resource_id,
            expected_device_model=model,
            expected_path=relative_path,
        )
        validated.append({
            "kind": resource.kind,
            "resource_id": resource.resource_id,
            "device_model": resource.device_model,
            "start_ms": resource.start_ms,
            "end_ms": resource.end_ms,
            "size_bytes": resource.size_bytes,
            "sha256": resource.sha256,
        })
    return validated


def _resources_for_validation(
    resource_metadata: object | None,
) -> dict[tuple[str, str], EvidenceResource]:
    if resource_metadata is None:
        return _load_manifest()
    return {
        (item["device_model"], item["kind"]): EvidenceResource(
            **item,
            relative_path=_EXPECTED_LAYOUT[item["device_model"]][item["kind"]][1],
        )
        for item in validate_resource_metadata(resource_metadata)
    }


def authoritative_tool_arguments(
    tool_name: str,
    device_model: str,
    *,
    resource_metadata: object | None = None,
) -> dict[str, Any]:
    """Materialize server-authoritative default arguments for one typed tool."""

    try:
        if tool_name not in SUPPORTED_TOOL_NAMES:
            _raise(TOOL_UNKNOWN)
        if not isinstance(device_model, str) or device_model not in SUPPORTED_DEVICE_MODELS:
            _raise(TOOL_SCOPE_INVALID)
        kind = "telemetry" if tool_name == SIMULATED_TELEMETRY_TOOL else "ros_log"
        resource = _resources_for_validation(resource_metadata).get((device_model, kind))
        if resource is None:
            _raise(TOOL_SCOPE_INVALID)
        if tool_name == SIMULATED_TELEMETRY_TOOL:
            return {
                "device_model": device_model,
                "resource_id": resource.resource_id,
                "start_ms": resource.start_ms,
                "end_ms": resource.end_ms,
                "max_points": MAX_TELEMETRY_POINTS,
                "max_bytes": MAX_EVIDENCE_BYTES,
            }
        return {
            "device_model": device_model,
            "resource_id": resource.resource_id,
            "start_ms": resource.start_ms,
            "end_ms": resource.end_ms,
            "max_entries": MAX_ROS_LOG_ENTRIES,
            "max_bytes": MAX_EVIDENCE_BYTES,
        }
    except EvidenceToolError:
        raise
    except Exception:
        raise EvidenceToolError(TOOL_EXECUTION_ERROR) from None


def normalize_evidence_tool_arguments(
    tool_name: str, arguments: object, *, resource_metadata: object | None = None
) -> dict[str, Any]:
    """Validate exact typed arguments against the server-owned manifest."""

    if not isinstance(tool_name, str) or tool_name not in SUPPORTED_TOOL_NAMES:
        _raise(TOOL_UNKNOWN)
    expected_keys = (
        {
            "device_model",
            "resource_id",
            "start_ms",
            "end_ms",
            "max_points",
            "max_bytes",
        }
        if tool_name == SIMULATED_TELEMETRY_TOOL
        else {
            "device_model",
            "resource_id",
            "start_ms",
            "end_ms",
            "max_entries",
            "max_bytes",
        }
    )
    if not isinstance(arguments, dict) or set(arguments) != expected_keys:
        _raise(TOOL_ARGUMENT_INVALID)
    device_model = arguments.get("device_model")
    if not isinstance(device_model, str):
        _raise(TOOL_SCOPE_INVALID)
    expected = authoritative_tool_arguments(
        tool_name, device_model, resource_metadata=resource_metadata
    )
    resource_id = arguments.get("resource_id")
    if not isinstance(resource_id, str) or _RESOURCE_ID_RE.fullmatch(resource_id) is None:
        _raise(TOOL_PATH_INVALID)
    if resource_id != expected["resource_id"]:
        _raise(TOOL_SCOPE_INVALID)
    start_ms = _strict_int(arguments.get("start_ms"), code=TOOL_TIME_WINDOW_INVALID)
    end_ms = _strict_int(arguments.get("end_ms"), code=TOOL_TIME_WINDOW_INVALID)
    if (
        start_ms >= end_ms
        or end_ms - start_ms > MAX_TIME_WINDOW_MS
        or start_ms < expected["start_ms"]
        or end_ms > expected["end_ms"]
    ):
        _raise(TOOL_TIME_WINDOW_INVALID)
    limit_name = (
        "max_points"
        if tool_name == SIMULATED_TELEMETRY_TOOL
        else "max_entries"
    )
    if (
        type(arguments.get(limit_name)) is not int
        or arguments[limit_name] != expected[limit_name]
    ):
        _raise(TOOL_ARGUMENT_INVALID)
    if (
        type(arguments.get("max_bytes")) is not int
        or arguments["max_bytes"] != MAX_EVIDENCE_BYTES
    ):
        _raise(TOOL_ARGUMENT_INVALID)
    return {
        "device_model": device_model,
        "resource_id": resource_id,
        "start_ms": start_ms,
        "end_ms": end_ms,
        limit_name: arguments[limit_name],
        "max_bytes": arguments["max_bytes"],
    }


# Backward-compatible public name plus the short name used by the tool loop.
validate_tool_arguments = normalize_evidence_tool_arguments


def _telemetry_sample(
    value: object, *, error_code: str = TOOL_INPUT_FORMAT_INVALID
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "ts_ms",
        "metric",
        "value",
        "unit",
        "quality",
    }:
        _raise(error_code)
    metric = value.get("metric")
    unit = value.get("unit")
    if not isinstance(metric, str) or metric not in _METRIC_UNITS:
        _raise(error_code)
    if unit != _METRIC_UNITS[metric]:
        _raise(error_code)
    quality = value.get("quality")
    if quality not in _TELEMETRY_QUALITIES:
        _raise(error_code)
    timestamp_ms = _strict_int(value.get("ts_ms"), code=error_code)
    number = value.get("value")
    if (
        isinstance(number, bool)
        or not isinstance(number, (int, float))
        or not math.isfinite(float(number))
    ):
        _raise(error_code)
    return {
        "ts_ms": timestamp_ms,
        "metric": metric,
        "value": float(number),
        "unit": unit,
        "quality": quality,
    }


def _ros_log_entry(
    value: object,
    *,
    error_code: str = TOOL_INPUT_FORMAT_INVALID,
    require_redacted: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "ts_ms",
        "level",
        "node",
        "message",
    }:
        _raise(error_code)
    level = value.get("level")
    if level not in _LOG_LEVELS:
        _raise(error_code)
    try:
        node = _bounded_safe_text(value.get("node"), maximum=128)
        message = _bounded_safe_text(
            value.get("message"), maximum=MAX_LOG_MESSAGE_CHARS
        )
    except EvidenceToolError:
        _raise(error_code)
    if _NODE_RE.fullmatch(node) is None:
        _raise(error_code)
    timestamp_ms = _strict_int(value.get("ts_ms"), code=error_code)
    redacted_message = _redact_sensitive(message)
    if require_redacted and message != redacted_message:
        _raise(error_code)
    return {
        "ts_ms": timestamp_ms,
        "level": level,
        "node": node,
        "message": redacted_message,
    }


def _resource_for(
    resources: Mapping[tuple[str, str], EvidenceResource],
    tool_name: str,
    device_model: str,
) -> EvidenceResource:
    kind = "telemetry" if tool_name == SIMULATED_TELEMETRY_TOOL else "ros_log"
    resource = resources.get((device_model, kind))
    if resource is None:
        _raise(TOOL_SCOPE_INVALID)
    return resource


def _assert_resource_binding(
    resource: EvidenceResource, arguments: Mapping[str, Any]
) -> None:
    if (
        resource.device_model != arguments.get("device_model")
        or resource.resource_id != arguments.get("resource_id")
    ):
        _raise(TOOL_SCOPE_INVALID)
    if (
        resource.start_ms > arguments.get("start_ms", -1)
        or resource.end_ms < arguments.get("end_ms", -1)
    ):
        _raise(TOOL_TIME_WINDOW_INVALID)
    if resource.size_bytes > arguments.get("max_bytes", -1):
        _raise(TOOL_INPUT_TOO_LARGE)


def normalize_tool_result(
    tool_name: str,
    output: object,
    arguments: object,
    *,
    resource_metadata: object | None = None,
) -> dict[str, Any]:
    """Revalidate a result against typed arguments and selected resource metadata.

    This intentionally does not reopen the evidence resource, so checkpoint recovery
    can validate its bound normalized result without repeating the tool call.
    """

    try:
        normalized_arguments = validate_tool_arguments(
            tool_name, arguments, resource_metadata=resource_metadata
        )
        resource = _resource_for(
            _resources_for_validation(resource_metadata),
            tool_name,
            normalized_arguments["device_model"],
        )
        _assert_resource_binding(resource, normalized_arguments)
        collection_key = (
            "samples" if tool_name == SIMULATED_TELEMETRY_TOOL else "entries"
        )
        expected_keys = {
            "schema_version",
            "source",
            "trust_boundary",
            "device_model",
            "resource_id",
            "start_ms",
            "end_ms",
            "bytes_read",
            "resource_sha256",
            "count",
            "truncated",
            collection_key,
        }
        if not isinstance(output, dict) or set(output) != expected_keys:
            _raise(TOOL_OUTPUT_INVALID)
        if (
            type(output.get("schema_version")) is not int
            or output["schema_version"] != RESULT_SCHEMA_VERSION
            or output.get("source") != RESULT_SOURCE
            or output.get("trust_boundary") != RESULT_TRUST_BOUNDARY
            or output.get("device_model") != normalized_arguments["device_model"]
            or output.get("resource_id") != normalized_arguments["resource_id"]
        ):
            _raise(TOOL_OUTPUT_INVALID)
        for name in ("start_ms", "end_ms", "bytes_read", "count"):
            if type(output.get(name)) is not int:
                _raise(TOOL_OUTPUT_INVALID)
        if (
            output["start_ms"] != normalized_arguments["start_ms"]
            or output["end_ms"] != normalized_arguments["end_ms"]
            or output["bytes_read"] != resource.size_bytes
            or output["bytes_read"] > normalized_arguments["max_bytes"]
            or not isinstance(output.get("resource_sha256"), str)
            or _SHA256_RE.fullmatch(output["resource_sha256"]) is None
            or not secrets.compare_digest(
                output["resource_sha256"], resource.sha256
            )
            or output["count"] < 0
            or type(output.get("truncated")) is not bool
            or not isinstance(output.get(collection_key), list)
        ):
            _raise(TOOL_OUTPUT_INVALID)

        if tool_name == SIMULATED_TELEMETRY_TOOL:
            records = [
                _telemetry_sample(item, error_code=TOOL_OUTPUT_INVALID)
                for item in output[collection_key]
            ]
            order_key = lambda item: (item["ts_ms"], item["metric"])
            identities = [(item["ts_ms"], item["metric"]) for item in records]
            maximum_count = normalized_arguments["max_points"]
        else:
            records = [
                _ros_log_entry(
                    item,
                    error_code=TOOL_OUTPUT_INVALID,
                    require_redacted=True,
                )
                for item in output[collection_key]
            ]
            order_key = lambda item: (
                item["ts_ms"],
                item["node"],
                item["level"],
                item["message"],
            )
            identities = [
                (item["ts_ms"], item["node"], item["level"], item["message"])
                for item in records
            ]
            maximum_count = normalized_arguments["max_entries"]

        if (
            output["count"] != len(records)
            or len(records) > maximum_count
            or (output["truncated"] and len(records) != maximum_count)
            or any(
                item["ts_ms"] < normalized_arguments["start_ms"]
                or item["ts_ms"] >= normalized_arguments["end_ms"]
                for item in records
            )
            or records != sorted(records, key=order_key)
            or len(identities) != len(set(identities))
        ):
            _raise(TOOL_OUTPUT_INVALID)

        normalized = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "source": RESULT_SOURCE,
            "trust_boundary": RESULT_TRUST_BOUNDARY,
            "device_model": normalized_arguments["device_model"],
            "resource_id": normalized_arguments["resource_id"],
            "start_ms": normalized_arguments["start_ms"],
            "end_ms": normalized_arguments["end_ms"],
            "bytes_read": output["bytes_read"],
            "resource_sha256": resource.sha256,
            "count": len(records),
            "truncated": output["truncated"],
            collection_key: records,
        }
        _canonical_json_bytes(normalized, error_code=TOOL_OUTPUT_INVALID)
        return normalized
    except EvidenceToolError:
        raise
    except Exception:
        raise EvidenceToolError(TOOL_OUTPUT_INVALID) from None


def query_simulated_telemetry_strict(
    device_model: str,
    resource_id: str,
    start_ms: int,
    end_ms: int,
    max_points: int,
    max_bytes: int,
) -> dict[str, Any]:
    """Read bounded synthetic telemetry; never reaches a robot or network."""

    try:
        arguments = validate_tool_arguments(
            SIMULATED_TELEMETRY_TOOL,
            {
                "device_model": device_model,
                "resource_id": resource_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "max_points": max_points,
                "max_bytes": max_bytes,
            },
        )
        resource = _resource_for(
            _load_manifest(), SIMULATED_TELEMETRY_TOOL, arguments["device_model"]
        )
        _assert_resource_binding(resource, arguments)
        payload, digest = _read_bounded_resource(
            resource.relative_path,
            maximum_bytes=arguments["max_bytes"],
            expected_size_bytes=resource.size_bytes,
            expected_sha256=resource.sha256,
        )
        samples = [_telemetry_sample(item) for item in _decode_jsonl(payload)]
        if any(
            item["ts_ms"] < resource.start_ms or item["ts_ms"] >= resource.end_ms
            for item in samples
        ):
            _raise(TOOL_TIME_WINDOW_INVALID)
        samples.sort(key=lambda item: (item["ts_ms"], item["metric"]))
        identities = [(item["ts_ms"], item["metric"]) for item in samples]
        if len(identities) != len(set(identities)):
            _raise(TOOL_INPUT_FORMAT_INVALID)
        filtered = [
            item
            for item in samples
            if arguments["start_ms"] <= item["ts_ms"] < arguments["end_ms"]
        ]
        truncated = len(filtered) > arguments["max_points"]
        filtered = filtered[: arguments["max_points"]]
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "source": RESULT_SOURCE,
            "trust_boundary": RESULT_TRUST_BOUNDARY,
            "device_model": arguments["device_model"],
            "resource_id": arguments["resource_id"],
            "start_ms": arguments["start_ms"],
            "end_ms": arguments["end_ms"],
            "bytes_read": len(payload),
            "resource_sha256": digest,
            "count": len(filtered),
            "truncated": truncated,
            "samples": filtered,
        }
        return normalize_tool_result(
            SIMULATED_TELEMETRY_TOOL, result, arguments
        )
    except EvidenceToolError:
        raise
    except Exception:
        raise EvidenceToolError(TOOL_EXECUTION_ERROR) from None


def analyze_synthetic_ros_log_strict(
    device_model: str,
    resource_id: str,
    start_ms: int,
    end_ms: int,
    max_entries: int,
    max_bytes: int,
) -> dict[str, Any]:
    """Parse one bounded synthetic ROS JSONL log as untrusted data only."""

    try:
        arguments = validate_tool_arguments(
            SYNTHETIC_ROS_LOG_TOOL,
            {
                "device_model": device_model,
                "resource_id": resource_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "max_entries": max_entries,
                "max_bytes": max_bytes,
            },
        )
        resource = _resource_for(
            _load_manifest(), SYNTHETIC_ROS_LOG_TOOL, arguments["device_model"]
        )
        _assert_resource_binding(resource, arguments)
        payload, digest = _read_bounded_resource(
            resource.relative_path,
            maximum_bytes=arguments["max_bytes"],
            expected_size_bytes=resource.size_bytes,
            expected_sha256=resource.sha256,
        )
        entries = [_ros_log_entry(item) for item in _decode_jsonl(payload)]
        if any(
            item["ts_ms"] < resource.start_ms or item["ts_ms"] >= resource.end_ms
            for item in entries
        ):
            _raise(TOOL_TIME_WINDOW_INVALID)
        entries.sort(
            key=lambda item: (
                item["ts_ms"],
                item["node"],
                item["level"],
                item["message"],
            )
        )
        identities = [
            (item["ts_ms"], item["node"], item["level"], item["message"])
            for item in entries
        ]
        if len(identities) != len(set(identities)):
            _raise(TOOL_INPUT_FORMAT_INVALID)
        filtered = [
            item
            for item in entries
            if arguments["start_ms"] <= item["ts_ms"] < arguments["end_ms"]
        ]
        truncated = len(filtered) > arguments["max_entries"]
        filtered = filtered[: arguments["max_entries"]]
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "source": RESULT_SOURCE,
            "trust_boundary": RESULT_TRUST_BOUNDARY,
            "device_model": arguments["device_model"],
            "resource_id": arguments["resource_id"],
            "start_ms": arguments["start_ms"],
            "end_ms": arguments["end_ms"],
            "bytes_read": len(payload),
            "resource_sha256": digest,
            "count": len(filtered),
            "truncated": truncated,
            "entries": filtered,
        }
        return normalize_tool_result(SYNTHETIC_ROS_LOG_TOOL, result, arguments)
    except EvidenceToolError:
        raise
    except Exception:
        raise EvidenceToolError(TOOL_EXECUTION_ERROR) from None


__all__ = [
    "EvidenceToolError",
    "MAX_EVIDENCE_BYTES",
    "MAX_ROS_LOG_ENTRIES",
    "MAX_TELEMETRY_POINTS",
    "MAX_TIME_WINDOW_MS",
    "RESULT_SCHEMA_VERSION",
    "RESULT_SOURCE",
    "RESULT_TRUST_BOUNDARY",
    "SIMULATED_TELEMETRY_TOOL",
    "STABLE_ERROR_CODES",
    "SUPPORTED_DEVICE_MODELS",
    "SUPPORTED_TOOL_NAMES",
    "SYNTHETIC_ROS_LOG_TOOL",
    "analyze_synthetic_ros_log_strict",
    "authoritative_tool_arguments",
    "normalize_evidence_tool_arguments",
    "normalize_tool_result",
    "query_simulated_telemetry_strict",
    "validate_tool_arguments",
]
