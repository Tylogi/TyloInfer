"""Repeatable, strict test gates for local and CI validation.

The quick gate intentionally covers only deterministic CPU/Python and browser
unit tests. The native gate is separate because it needs a configured build,
and refuses missing resources, empty CTest selections, or skipped tests.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from datetime import timezone
from xml.etree import ElementTree
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NATIVE_TARGETS = (
    "tyloinfer-numeric-assertions-test",
    "mfq-flash-next-test",
    "mfq-cuda-runtime-state-test",
    "tyloinfer-cuda-tensor-capture-lifetime-test",
)
DEFAULT_NATIVE_REGEX = (
    "^(tyloinfer-numeric-assertions-test|tyloinfer-cuda-numeric-assertions-test|"
    "tyloinfer-cuda-attention-prefix-test|tyloinfer-cuda-tensor-capture-lifetime-test|"
    "mfq-flash-next-test)$"
)
DEFAULT_NATIVE_TESTS = (
    "tyloinfer-numeric-assertions-test",
    "tyloinfer-cuda-numeric-assertions-test",
    "tyloinfer-cuda-attention-prefix-test",
    "tyloinfer-cuda-tensor-capture-lifetime-test",
    "mfq-flash-next-test",
)
RUNTIME_TEST_FUNCTIONS = (
    "test_native_stdio_loads_and_generates",
    "test_native_stdio_stream_matches_non_stream",
    "test_native_stdio_rejects_bad_request_and_recovers",
    "test_native_stdio_cancel_and_reload_recover",
    "test_native_stdio_mtp_matches_plain_when_available",
    "test_native_stdio_json_schema_output",
)
RUNTIME_CASES = tuple(
    f"{function}[{batch}]"
    for function in RUNTIME_TEST_FUNCTIONS
    for batch in ("0", "2")
) + (
    "test_native_stdio_production_client_matches_reference_oracle",
    "test_native_stdio_production_client_multiplexes_and_recovers",
)
RUNTIME_MTP_CASES = {
    case for case in RUNTIME_CASES if "mtp_matches_plain_when_available" in case
}
CPU_REQUIRED_TESTS = {
    "tyloinfer-numeric-assertions-test",
    "mfq-native-tensor-test",
    "mfq-model-source-test",
    "mfq-file-range-reader-test",
    "mfq-prefill-budget-test",
    "mfq-speculative-commit-test",
}
COMMUNICATION_REQUIRED_TESTS = {
    "mfq-tokenizer-test",
    "mfq-tokenizer-identity-test",
    "mfq-text-processor-test",
    "mfq-inference-test",
    "mfq-request-executor-test",
    "mfq-shared-execution-test",
    "mfq-speculative-sequence-test",
    "tyloinfer-session-cache-restore-test",
    "mfq-paged-prefix-cache-test",
    "mfq-decision-test",
    "mfq-runtime-transport-test",
    "mfq-runtime-profile-test",
}


def _git_value(*args: str) -> str:
    """Return a git value for the run record without making git mandatory."""
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _run(
    command: Sequence[str],
    *,
    label: str,
    log: list[str],
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> int:
    """Run one gate command, echoing and retaining its complete output."""
    rendered = " ".join(command)
    log.append(f"\n$ {rendered}\n")
    try:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as error:
        log.append(f"{label}: unable to start command: {error}\n")
        return 127
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        log.append(line)
    return process.wait()


def _tool(name: str, fallback: str | None = None) -> str:
    """Resolve a CLI on POSIX and Windows, including Windows .cmd shims."""
    configured = os.environ.get(f"TYLOINFER_{name.upper()}")
    if configured:
        return configured
    resolved = shutil.which(name)
    if resolved:
        return resolved
    if os.name == "nt":
        resolved = shutil.which(f"{name}.cmd") or shutil.which(f"{name}.exe")
        if resolved:
            return resolved
    return fallback or name


def _write_record(name: str, record: dict[str, object], log: list[str]) -> Path:
    """Write a timestamped log and JSON run record under the ignored local folder."""
    output = ROOT / ".tyloinfer"
    output.mkdir(exist_ok=True)
    stamp = dt.datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_path = output / f"test-gate-{name}-{stamp}.log"
    record_path = output / f"test-gate-{name}-{stamp}.json"
    log_path.write_text("".join(log), encoding="utf-8")
    record["log"] = str(log_path.relative_to(ROOT))
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"gate record: {record_path.relative_to(ROOT)}")
    return record_path


def _base_record(name: str) -> dict[str, object]:
    return {
        "gate": name,
        "timestamp_utc": dt.datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "platform": sys.platform,
        "python": sys.version.split()[0],
    }


def _asset_record(value: str) -> dict[str, object]:
    """Record resolved asset provenance without hashing potentially huge weights."""
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    record: dict[str, object] = {"path": str(path.resolve()), "exists": path.exists()}
    if path.exists():
        stat = path.stat()
        record.update(
            {
                "kind": "directory" if path.is_dir() else "file",
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return record


def _cmake_cache_record(build_dir: Path) -> dict[str, str]:
    """Keep the toolchain/backend settings that explain a native gate result."""
    cache = build_dir / "CMakeCache.txt"
    keys = (
        "CMAKE_BUILD_TYPE",
        "CMAKE_CXX_COMPILER",
        "CMAKE_CUDA_COMPILER",
        "CMAKE_CUDA_ARCHITECTURES",
        "CMAKE_GENERATOR",
        "TYLOINFER_BUILD_CUDA_RUNTIME",
        "TYLOINFER_BUILD_HIP_RUNTIME",
        "TYLOINFER_BUILD_METAL_RUNTIME",
        "TYLOINFER_BUILD_RUNTIME_COMMUNICATION",
    )
    if not cache.exists():
        return {}
    values: dict[str, str] = {}
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("//") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.split(":", 1)[0]
        if key in keys:
            values[key] = value
    return values


def run_quick(_: argparse.Namespace) -> int:
    """Run deterministic Python and frontend unit/type checks."""
    log: list[str] = []
    record = _base_record("quick")
    uv = shutil.which("uv") or shutil.which("uv.exe")
    if uv:
        python_command = [
            uv,
            "run",
            "--locked",
            "--project",
            "studio",
            "--extra",
            "dev",
            "pytest",
            "studio/tests",
            "-m",
            "not real_runtime",
            "-ra",
        ]
    else:
        # A provisioned project venv keeps the gate usable on Windows where uv
        # may not be on PATH.
        python = os.environ.get("TYLOINFER_TEST_PYTHON", sys.executable)
        if python == sys.executable:
            for candidate in (
                ROOT / ".venv" / "Scripts" / "python.exe",
                ROOT / ".venv" / "bin" / "python",
            ):
                if candidate.exists():
                    python = str(candidate)
                    break
        python_command = [
            python,
            "-m",
            "pytest",
            "studio/tests",
            "-m",
            "not real_runtime",
            "-ra",
        ]
    pnpm = _tool("pnpm")
    commands = [
        ("python", python_command),
        ("frontend-unit", [pnpm, "--dir", "studio/web", "run", "test:unit"]),
        ("frontend-typecheck", [pnpm, "--dir", "studio/web", "run", "typecheck"]),
    ]
    results: dict[str, int] = {}
    for label, command in commands:
        results[label] = _run(command, label=label, log=log)
    cpu_results, cpu_records = _run_cpu_profiles(log)
    results.update(cpu_results)
    plugin_code, plugin_records = _run_plugins(log)
    results["plugins"] = plugin_code
    record["commands"] = [command for _, command in commands]
    record["results"] = results
    record["cpu_native"] = cpu_records
    record["plugins"] = plugin_records
    status = 0 if all(code == 0 for code in results.values()) else 1
    record["status"] = "passed" if status == 0 else "failed"
    _write_record("quick", record, log)
    return status


def _test_count(discovery: str) -> int:
    return len(re.findall(r"^\s*Test\s+#\d+:", discovery, re.MULTILINE))


def _test_names(discovery: str) -> tuple[str, ...]:
    return tuple(re.findall(r"^\s*Test\s+#\d+:\s*(\S+)", discovery, re.MULTILINE))


def _has_skip(output: str) -> bool:
    if re.search(r"Skipped Tests:\s*[1-9]\d*", output, re.IGNORECASE):
        return True
    return bool(re.search(r"\(Not Run\)|\*{3}\s*Skipped\b", output, re.IGNORECASE))


def _run_strict_ctest(
    ctest: str,
    build_dir: Path,
    regex: str | None,
    log: list[str],
    config: str | None = None,
) -> tuple[int, int, int, bool, tuple[str, ...]]:
    """Discover and run CTest once, returning status and discovered names."""
    discovery_command = [ctest, "--test-dir", str(build_dir), "-N"]
    if config:
        discovery_command.extend(["-C", config])
    if regex:
        discovery_command.extend(["-R", regex])
    log.append(f"\n$ {' '.join(discovery_command)}\n")
    try:
        discovery_process = subprocess.run(
            discovery_command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        discovery = discovery_process.stdout + discovery_process.stderr
        discovery_code = discovery_process.returncode
    except OSError as error:
        discovery = str(error)
        discovery_code = 127
    sys.stdout.write(discovery)
    log.append(discovery)
    count = _test_count(discovery)
    names = _test_names(discovery)
    if discovery_code or count == 0:
        return (discovery_code or 3), count, 0, False, names

    run_command = [
        ctest,
        "--test-dir",
        str(build_dir),
        "--no-tests=error",
        "--output-on-failure",
    ]
    if config:
        run_command.extend(["-C", config])
    if regex:
        run_command.extend(["-R", regex])
    ctest_log_start = len(log)
    run_code = _run(run_command, label="ctest", log=log)
    ctest_output = "".join(log[ctest_log_start:])
    skipped = _has_skip(ctest_output)
    summary = re.search(
        r"(\d+)%\s+tests?\s+passed,\s*(\d+)\s+tests?\s+failed\s+out\s+of\s+(\d+)",
        ctest_output,
        re.IGNORECASE,
    )
    if summary:
        failed = int(summary.group(2))
        total = int(summary.group(3))
        passed = total - failed
    else:
        passed_match = re.search(
            r"(\d+)\s+tests?\s+passed", ctest_output, re.IGNORECASE
        )
        passed = int(passed_match.group(1)) if passed_match else 0
    complete = run_code == 0 and not skipped and passed == count
    return (0 if complete else (run_code or 4)), count, passed, skipped, names


def _run_cpu_native(
    log: list[str], *, communication: bool = False
) -> tuple[int, dict[str, object]]:
    """Build one CPU profile and require its important behavior tests to exist."""
    cmake = _tool("cmake")
    ctest = _tool("ctest")
    variable = "TYLOINFER_CPU_COMM_BUILD_DIR" if communication else "TYLOINFER_CPU_BUILD_DIR"
    configured_build_dir = os.environ.get(variable)
    build_dir = (
        Path(configured_build_dir)
        if configured_build_dir
        else ROOT / ".tyloinfer" / ("test-gate-cpu-comm-build" if communication else "test-gate-cpu-build")
    )
    if not build_dir.is_absolute():
        build_dir = ROOT / build_dir
    configure = [
        cmake,
        "-S",
        str(ROOT / "csrc"),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_TESTING=ON",
        "-DTYLOINFER_BUILD_CUDA_RUNTIME=OFF",
        "-DTYLOINFER_BUILD_HIP_RUNTIME=OFF",
        "-DTYLOINFER_BUILD_METAL_RUNTIME=OFF",
        f"-DTYLOINFER_BUILD_RUNTIME_COMMUNICATION={'ON' if communication else 'OFF'}",
    ]
    if not (build_dir / "CMakeCache.txt").exists() and (
        shutil.which("ninja") or shutil.which("ninja.exe")
    ):
        configure[1:1] = ["-G", "Ninja"]
    result: dict[str, object] = {
        "build_dir": str(build_dir),
        "configuration": "Release",
        "communication": communication,
        "options": configure[configure.index("-DCMAKE_BUILD_TYPE=Release") :],
    }
    configure_code = _run(configure, label="cpu-configure", log=log)
    if configure_code:
        result.update({"status": "failed", "failure": "CPU CMake configure failed"})
        return configure_code, result
    build_code = _run(
        [cmake, "--build", str(build_dir), "--config", "Release", "--parallel"],
        label="cpu-build",
        log=log,
    )
    if build_code:
        result.update({"status": "failed", "failure": "CPU CMake build failed"})
        return build_code, result
    ctest_code, count, passed, skipped, names = _run_strict_ctest(
        ctest, build_dir, None, log, "Release"
    )
    required = CPU_REQUIRED_TESTS | (COMMUNICATION_REQUIRED_TESTS if communication else set())
    missing = sorted(required - set(names))
    if missing:
        ctest_code = ctest_code or 5
    result.update(
        {
            "discovered_tests": count,
            "passed_tests": passed,
            "skipped_tests": skipped,
            "required_tests": sorted(required),
            "missing_tests": missing,
            "discovered_names": names,
            "status": "passed" if ctest_code == 0 else "failed",
        }
    )
    if ctest_code:
        result["failure"] = "CPU CTest suite was empty, missing a required test, failed, or skipped"
    return ctest_code, result


def _run_plugins(log: list[str]) -> tuple[int, dict[str, object]]:
    """Build the Client and run every Host/Client Node test with strict TAP results."""
    pnpm = _tool("pnpm")
    records: dict[str, object] = {}
    failed = False
    for package, required_file in (
        ("host", "model-actions.test.mjs"),
        ("studio", "model-management.test.mjs"),
    ):
        directory = ROOT / "plugins" / package
        files = sorted((directory / "tests").glob("*.test.mjs"))
        result: dict[str, object] = {"files": [path.name for path in files]}
        records[package] = result
        if not files or required_file not in result["files"]:
            result.update(status="failed", failure="required plugin test files are missing")
            failed = True
            continue
        if package == "studio":
            builds = [
                [pnpm, "--dir", str(directory), "run", "build"],
                [pnpm, "--dir", str(directory), "run", "build", "--dsbot"],
            ]
            result["build_commands"] = builds
            for build in builds:
                if _run(build, label="plugin-studio-build", log=log):
                    result.update(status="failed", failure="plugin build failed")
                    failed = True
                    break
            if result.get("status") == "failed":
                continue
        command = [
            pnpm, "--dir", str(directory), "exec", "node", "--test", "--test-reporter=tap",
            *(str(path.relative_to(directory)) for path in files),
        ]
        result["command"] = command
        start = len(log)
        environment = {**os.environ, "TYLOINFER_TEST_DSBOT": "1"}
        code = _run(command, label=f"plugin-{package}", log=log, env=environment)
        output = "".join(log[start:])
        counts = {
            key: int(matches[-1])
            for key in ("tests", "pass", "fail", "cancelled", "skipped", "todo")
            if (matches := re.findall(rf"^# {key} (\d+)\s*$", output, re.MULTILINE))
        }
        valid = (
            code == 0 and len(counts) == 6 and counts["tests"] > 0
            and counts["pass"] == counts["tests"]
            and not any(counts[key] for key in ("fail", "cancelled", "skipped", "todo"))
        )
        result.update(counts=counts, status="passed" if valid else "failed")
        failed = failed or not valid
    return (1 if failed else 0), records


def _run_cpu_profiles(log: list[str]) -> tuple[dict[str, int], dict[str, object]]:
    """Run both feature configurations so conditional registrations are exercised."""
    results: dict[str, int] = {}
    records: dict[str, object] = {}
    for communication in (False, True):
        profile = "communication-on" if communication else "communication-off"
        results[f"cpu-{profile}"], records[profile] = _run_cpu_native(
            log, communication=communication
        )
    return results, records


def run_cpu(_: argparse.Namespace) -> int:
    """Run both CPU feature profiles independently of Python and frontend suites."""
    log: list[str] = []
    record = _base_record("cpu")
    results, record["profiles"] = _run_cpu_profiles(log)
    code = 0 if all(value == 0 for value in results.values()) else 1
    record.update(results=results, status="passed" if code == 0 else "failed")
    _write_record("cpu", record, log)
    return code


def run_plugins(_: argparse.Namespace) -> int:
    """Run the two repository plugin suites without Python or accelerator requirements."""
    log: list[str] = []
    record = _base_record("plugins")
    code, record["plugins"] = _run_plugins(log)
    record["status"] = "passed" if code == 0 else "failed"
    _write_record("plugins", record, log)
    return code


def run_native(args: argparse.Namespace) -> int:
    """Build selected native targets, then require every selected CTest to run."""
    build_dir = Path(args.build_dir)
    if not build_dir.is_absolute():
        build_dir = ROOT / build_dir
    targets = tuple(args.target or DEFAULT_NATIVE_TARGETS)
    resources = [Path(value) for value in args.require_resource]
    log: list[str] = []
    record = _base_record("native")
    record.update(
        {
            "build_dir": str(build_dir),
            "targets": list(targets),
            "ctest_regex": args.regex,
            "required_resources": [str(path) for path in resources],
        }
    )
    missing = [
        path
        for path in resources
        if not (path if path.is_absolute() else ROOT / path).exists()
    ]
    if missing:
        record["status"] = "failed"
        record["failure"] = f"required resource missing: {missing[0]}"
        _write_record("native", record, log)
        print(record["failure"], file=sys.stderr)
        return 2
    build_command = [
        args.cmake,
        "--build",
        str(build_dir),
        "--target",
        *targets,
        "--parallel",
    ]
    if args.config:
        build_command.extend(["--config", args.config])
    build_code = _run(build_command, label="native-build", log=log)
    if build_code:
        record["status"] = "failed"
        record["failure"] = "native build failed"
        _write_record("native", record, log)
        return build_code
    record["cmake_cache"] = _cmake_cache_record(build_dir)
    ctest_code, count, passed, skipped, names = _run_strict_ctest(
        args.ctest, build_dir, args.regex, log, args.config
    )
    record["discovered_tests"] = count
    record["passed_tests"] = passed
    record["skipped_tests"] = skipped
    record["discovered_names"] = list(names)
    required_tests = (
        tuple(args.required_test)
        if args.required_test is not None
        else DEFAULT_NATIVE_TESTS
        if args.regex == DEFAULT_NATIVE_REGEX
        else ()
    )
    expected = (
        args.expected
        if args.expected is not None
        else len(DEFAULT_NATIVE_TESTS)
        if args.regex == DEFAULT_NATIVE_REGEX
        else None
    )
    if required_tests and set(required_tests) != set(names):
        ctest_code = ctest_code or 5
        record["failure"] = "CTest discovery did not match required test names"
    if expected is not None and count != expected:
        ctest_code = ctest_code or 5
        record["failure"] = "CTest discovery count did not match the expected gate set"
    if ctest_code and count == 0:
        record["status"] = "failed"
        record["failure"] = "CTest selection is empty or discovery failed"
        _write_record("native", record, log)
        print(record["failure"], file=sys.stderr)
        return 3
    complete = ctest_code == 0
    record["status"] = "passed" if complete else "failed"
    if skipped:
        record["failure"] = "selected CTest run contained a skipped/not-run test"
    elif passed != count:
        record["failure"] = "CTest passed-test count did not match discovery"
    _write_record("native", record, log)
    return 0 if complete else ctest_code


def run_service(args: argparse.Namespace) -> int:
    """Run the repository's fixed real-service browser protocol entry point."""
    log: list[str] = []
    record = _base_record("service")
    command = [
        args.pnpm,
        "--dir",
        "studio/web",
        "exec",
        "playwright",
        "test",
        "--config",
        "e2e-service/playwright.config.ts",
        "--reporter=json",
    ]
    record.update(
        {
            "command": command,
            "expected_tests": args.expected,
        }
    )
    report_path = (
        ROOT
        / ".tyloinfer"
        / (
            "playwright-service-"
            f"{dt.datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
        )
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    # Playwright's JSON reporter writes the complete tree to this unique file.
    # Keeping it out of stdout avoids mistaking nested suites for the root report.
    environment["PLAYWRIGHT_JSON_OUTPUT_FILE"] = str(report_path)
    record["playwright_report"] = str(report_path.relative_to(ROOT))
    code = _run(command, label="service", log=log, env=environment)
    report = _read_playwright_report(report_path)
    passed, skipped, failed = _playwright_counts(report)
    valid = (
        code == 0
        and report is not None
        and passed == args.expected
        and skipped == failed == 0
    )
    record.update({"passed_tests": passed, "skipped_tests": skipped})
    record["failed_tests"] = failed
    record["status"] = "passed" if valid else "failed"
    if not valid:
        record["failure"] = "real-service browser entry was empty, failed, or skipped"
    _write_record("service", record, log)
    return 0 if valid else (code or 4)


def _read_playwright_report(path: Path) -> object | None:
    """Read one complete Playwright JSON report, never a nested suite fragment."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _playwright_counts(report: object) -> tuple[int, int, int]:
    """Count final Playwright spec statuses from its JSON reporter object."""
    if not isinstance(report, dict):
        return 0, 0, 0
    passed = skipped = failed = 0
    stack = list(report.get("suites", []))
    while stack:
        suite = stack.pop()
        stack.extend(suite.get("suites", []))
        for spec_result in suite.get("specs", []):
            tests = spec_result.get("tests", [])
            if not tests:
                failed += 1
            for test_result in tests:
                results = test_result.get("results", [])
                status = results[-1].get("status") if results else "failed"
                if status == "passed":
                    passed += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    failed += 1
    return passed, skipped, failed


def _runtime_junit_cases(path: Path) -> dict[str, str]:
    """Read pytest JUnit case identities and final statuses."""
    try:
        root = ElementTree.parse(path).getroot()
    except (ElementTree.ParseError, OSError):
        return {}
    cases: dict[str, str] = {}
    for testcase in root.iter("testcase"):
        name = testcase.attrib.get("name", "")
        if not name or name in cases:
            return {}
        if testcase.find("error") is not None:
            status = "error"
        elif testcase.find("failure") is not None:
            status = "failed"
        elif testcase.find("skipped") is not None:
            status = "skipped"
        else:
            status = "passed"
        cases[name] = status
    return cases


def _validate_runtime_cases(
    cases: dict[str, str], *, optional_mtp: bool
) -> dict[str, object]:
    """Validate the exact parameterized real-runtime set and allowed MTP skips."""
    expected_names = set(RUNTIME_CASES)
    expected = len(expected_names)
    identity_ok = set(cases) == expected_names
    failed = sorted(
        name for name, status in cases.items() if status in {"failed", "error"}
    )
    skipped = sorted(name for name, status in cases.items() if status == "skipped")
    invalid_skips = [
        name for name in skipped if not optional_mtp or name not in RUNTIME_MTP_CASES
    ]
    allowed_skips = [name for name in skipped if name in RUNTIME_MTP_CASES]
    passed = sorted(name for name, status in cases.items() if status == "passed")
    valid = identity_ok and not failed and not invalid_skips
    if optional_mtp:
        valid = valid and len(allowed_skips) <= len(RUNTIME_MTP_CASES)
    else:
        valid = valid and not skipped
    return {
        "identity_ok": identity_ok,
        "expected_tests": expected,
        "passed_tests": len(passed),
        "skipped_tests": len(skipped),
        "failed_tests": len(failed),
        "allowed_skipped_tests": allowed_skips,
        "invalid_skipped_tests": invalid_skips,
        "failed_case_names": failed,
        "cases": cases,
        "valid": valid,
    }


def run_runtime(args: argparse.Namespace) -> int:
    """Run the configured real-runtime pytest set with no allowed skips."""
    if args.require_mtp and args.optional_mtp:
        raise SystemExit("--require-mtp and --optional-mtp are mutually exclusive")
    log: list[str] = []
    record = _base_record("real-runtime")
    names = (
        "TYLOINFER_STEP_TEST_RUNTIME",
        "TYLOINFER_STEP_TEST_MODEL",
        "TYLOINFER_STEP_TEST_TOKENIZER",
    )
    configured_assets = {name: os.environ.get(name, "") for name in names}
    record["assets"] = {
        name: _asset_record(value) if value else {"path": "", "exists": False}
        for name, value in configured_assets.items()
    }
    manifest = os.environ.get("TYLOINFER_TEST_ASSET_MANIFEST")
    if manifest:
        manifest_record = _asset_record(manifest)
        manifest_path = Path(str(manifest_record["path"]))
        if manifest_path.is_file():
            manifest_record["sha256"] = hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()
        record["asset_manifest"] = manifest_record
    missing = []
    for name in names:
        value = configured_assets[name]
        path = Path(value) if value else None
        if path is None or not (path if path.is_absolute() else ROOT / path).exists():
            missing.append(name)
    if missing:
        record["status"] = "failed"
        record["failure"] = (
            f"real-runtime resource configuration missing: {', '.join(missing)}"
        )
        _write_record("real-runtime", record, log)
        print(record["failure"], file=sys.stderr)
        return 2
    environment = os.environ.copy()
    environment["TYLOINFER_REQUIRE_REAL_RUNTIME"] = "1"
    if args.require_mtp:
        environment["TYLOINFER_STEP_TEST_MTP"] = "1"
    uv = shutil.which("uv") or shutil.which("uv.exe")
    if uv:
        command = [
            uv,
            "run",
            "--locked",
            "--project",
            "studio",
            "--extra",
            "dev",
            "pytest",
            "studio/tests/test_native_stdio_e2e.py",
            "-m",
            "real_runtime",
            "-ra",
        ]
    else:
        python = _tool("test_python", sys.executable)
        command = [
            python,
            "-m",
            "pytest",
            "studio/tests/test_native_stdio_e2e.py",
            "-m",
            "real_runtime",
            "-ra",
        ]
    xml_path = (
        ROOT
        / ".tyloinfer"
        / (
            f"real-runtime-{dt.datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.xml"
        )
    )
    xml_path.parent.mkdir(exist_ok=True)
    command.append(f"--junitxml={xml_path}")
    record["command"] = command
    record["require_mtp"] = args.require_mtp
    record["optional_mtp"] = args.optional_mtp
    record["junitxml"] = str(xml_path.relative_to(ROOT))
    code = _run(command, label="real-runtime", log=log, env=environment)
    validation = _validate_runtime_cases(
        _runtime_junit_cases(xml_path),
        optional_mtp=args.optional_mtp,
    )
    record.update(validation)
    valid = code == 0 and bool(validation["valid"])
    record["status"] = "passed" if valid else "failed"
    if not valid:
        record["failure"] = (
            "real-runtime pytest set was empty, incomplete, failed, or skipped"
        )
    _write_record("real-runtime", record, log)
    return 0 if valid else (code or 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="gate", required=True)
    subparsers.add_parser("quick", help="run CPU/Python and frontend unit gates")
    subparsers.add_parser("plugins", help="run every Host and Client Node test")
    subparsers.add_parser("cpu", help="build and test communication OFF and ON CPU profiles")
    native = subparsers.add_parser("native", help="run a strict selected CTest gate")
    native.add_argument("--build-dir", default="build")
    native.add_argument("--cmake", default=os.environ.get("CMAKE", "cmake"))
    native.add_argument("--ctest", default=os.environ.get("CTEST", "ctest"))
    native.add_argument(
        "--config", help="multi-configuration build configuration, e.g. Release"
    )
    native.add_argument("--regex", default=DEFAULT_NATIVE_REGEX)
    native.add_argument("--expected", type=int)
    native.add_argument("--required-test", action="append")
    native.add_argument("--target", action="append")
    native.add_argument("--require-resource", action="append", default=[])
    service = subparsers.add_parser(
        "service", help="run the real-service browser protocol gate"
    )
    service.add_argument("--pnpm", default=_tool("pnpm"))
    service.add_argument("--expected", type=int, default=4)
    runtime = subparsers.add_parser(
        "real-runtime", help="run strict real native runtime pytest cases"
    )
    runtime.add_argument("--require-mtp", action="store_true")
    runtime.add_argument(
        "--optional-mtp",
        action="store_true",
        help="allow only the two MTP parameter cases to skip; not a zero-skip gate",
    )
    return parser.parse_args()


def main() -> int:
    # Preserve complete UTF-8 logs even when a legacy console cannot display
    # every character emitted by the test tools.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    args = parse_args()
    if args.gate == "quick":
        return run_quick(args)
    if args.gate == "native":
        return run_native(args)
    if args.gate == "service":
        return run_service(args)
    if args.gate == "plugins":
        return run_plugins(args)
    if args.gate == "cpu":
        return run_cpu(args)
    return run_runtime(args)


if __name__ == "__main__":
    raise SystemExit(main())
