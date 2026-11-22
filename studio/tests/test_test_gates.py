"""Regression tests for the strict test-gate entry points."""

from __future__ import annotations

import json
from argparse import Namespace
from collections.abc import Mapping
from pathlib import Path
from xml.etree import ElementTree

import pytest
from scripts import test_gates

# Keep this fixture independent from test_gates.RUNTIME_CASES: the gate must
# fail if either the producer or its expected-case list is accidentally changed.
EXPECTED_RUNTIME_CASES = (
    "test_native_stdio_loads_and_generates[0]",
    "test_native_stdio_loads_and_generates[2]",
    "test_native_stdio_stream_matches_non_stream[0]",
    "test_native_stdio_stream_matches_non_stream[2]",
    "test_native_stdio_rejects_bad_request_and_recovers[0]",
    "test_native_stdio_rejects_bad_request_and_recovers[2]",
    "test_native_stdio_cancel_and_reload_recover[0]",
    "test_native_stdio_cancel_and_reload_recover[2]",
    "test_native_stdio_mtp_matches_plain_when_available[0]",
    "test_native_stdio_mtp_matches_plain_when_available[2]",
    "test_native_stdio_json_schema_output[0]",
    "test_native_stdio_json_schema_output[2]",
    "test_native_stdio_production_client_matches_reference_oracle",
    "test_native_stdio_production_client_multiplexes_and_recovers",
)
MTP_CASES = (
    "test_native_stdio_mtp_matches_plain_when_available[0]",
    "test_native_stdio_mtp_matches_plain_when_available[2]",
)
NON_MTP_CASES = (
    "test_native_stdio_loads_and_generates[0]",
    "test_native_stdio_loads_and_generates[2]",
)


def _junit_report(statuses: Mapping[str, str] | None) -> str:
    root = ElementTree.Element("testsuite")
    for name, status in (statuses or {}).items():
        testcase = ElementTree.SubElement(root, "testcase", name=name)
        if status in {"failed", "error"}:
            tag = "failure" if status == "failed" else "error"
            ElementTree.SubElement(testcase, tag).text = "fixture failure"
        elif status == "skipped":
            ElementTree.SubElement(testcase, "skipped")
    return ElementTree.tostring(root, encoding="unicode")


def _fake_external_run(report: str, return_code: int, calls: list[list[str]]):
    def run(command, *, label, log, cwd=test_gates.ROOT, env=None):
        del label, log, cwd, env
        calls.append(list(command))
        xml_path = next(
            Path(argument.removeprefix("--junitxml="))
            for argument in command
            if argument.startswith("--junitxml=")
        )
        xml_path.write_text(report, encoding="utf-8")
        return return_code

    return run


@pytest.mark.parametrize(
    ("name", "optional_mtp", "mode", "command_code", "expected_code"),
    (
        ("all-pass-strict", False, "pass", 0, 0),
        ("all-pass-optional", True, "pass", 0, 0),
        ("one-mtp-skip-optional", True, "one-mtp-skip", 0, 0),
        ("two-mtp-skips-optional", True, "two-mtp-skips", 0, 0),
        ("one-mtp-skip-strict", False, "one-mtp-skip", 0, 4),
        ("two-non-mtp-skips", True, "two-non-mtp-skips", 0, 4),
        ("missing-case", True, "missing", 0, 4),
        ("missing-oracle", True, "missing-oracle", 0, 4),
        ("failed-case", True, "failed", 0, 4),
        ("error-case", True, "error", 0, 4),
        ("empty-report", True, "empty", 0, 4),
        ("external-command-failed", True, "pass", 7, 7),
    ),
)
def test_run_runtime_validates_real_junit_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    optional_mtp: bool,
    mode: str,
    command_code: int,
    expected_code: int,
) -> None:
    del name
    monkeypatch.setattr(test_gates, "ROOT", tmp_path)
    for asset_name in (
        "TYLOINFER_STEP_TEST_RUNTIME",
        "TYLOINFER_STEP_TEST_MODEL",
        "TYLOINFER_STEP_TEST_TOKENIZER",
    ):
        asset = tmp_path / asset_name.lower()
        if asset_name == "TYLOINFER_STEP_TEST_MODEL":
            asset.mkdir()
        else:
            asset.write_text("fixture", encoding="utf-8")
        monkeypatch.setenv(asset_name, str(asset))

    statuses = {case: "passed" for case in EXPECTED_RUNTIME_CASES}
    if mode == "one-mtp-skip":
        statuses[MTP_CASES[0]] = "skipped"
    elif mode == "two-mtp-skips":
        for case in MTP_CASES:
            statuses[case] = "skipped"
    elif mode == "two-non-mtp-skips":
        for case in NON_MTP_CASES:
            statuses[case] = "skipped"
    elif mode == "missing":
        statuses.pop(EXPECTED_RUNTIME_CASES[-1])
    elif mode == "missing-oracle":
        statuses.pop("test_native_stdio_production_client_matches_reference_oracle")
    elif mode == "error":
        statuses[EXPECTED_RUNTIME_CASES[0]] = "error"
    elif mode == "failed":
        statuses[EXPECTED_RUNTIME_CASES[0]] = "failed"
    report = "<testsuite/>" if mode == "empty" else _junit_report(statuses)
    calls: list[list[str]] = []
    monkeypatch.setattr(test_gates, "_run", _fake_external_run(report, command_code, calls))

    result = test_gates.run_runtime(
        Namespace(require_mtp=False, optional_mtp=optional_mtp)
    )

    assert result == expected_code
    assert len(calls) == 1


def test_playwright_counts_counts_each_project_execution() -> None:
    report = {
        "suites": [
            {
                "specs": [
                    {
                        "title": "one spec",
                        "tests": [
                            {"projectName": "chromium", "results": [{"status": "skipped"}]},
                            {"projectName": "firefox", "results": [{"status": "passed"}]},
                        ],
                    }
                ]
            }
        ]
    }

    assert test_gates._playwright_counts(report) == (1, 1, 0)


def test_service_reads_one_complete_nested_playwright_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nested describe suites and stdout noise must not change report selection."""
    monkeypatch.setattr(test_gates, "ROOT", tmp_path)
    calls: list[dict[str, object]] = []
    report = {
        "suites": [
            {
                "title": "file suite",
                "suites": [
                    {
                        "title": "describe suite",
                        "specs": [
                            {
                                "title": "nested one",
                                "tests": [{"results": [{"status": "passed"}]}],
                            },
                            {
                                "title": "nested two",
                                "tests": [{"results": [{"status": "passed"}]}],
                            },
                        ],
                    }
                ],
            }
        ]
    }

    def fake_run(command, *, label, log, env=None, **kwargs):
        del command, label, kwargs
        assert env is not None
        report_path = Path(env["PLAYWRIGHT_JSON_OUTPUT_FILE"])
        report_path.write_text(json.dumps(report), encoding="utf-8")
        log.append('{"suites":[{"specs":[]}]}\n')
        calls.append({"env": env, "report": report_path})
        return 0

    monkeypatch.setattr(test_gates, "_run", fake_run)
    result = test_gates.run_service(Namespace(pnpm="pnpm", expected=2))

    assert result == 0
    assert len(calls) == 1
    report_path = calls[0]["report"]
    assert isinstance(report_path, Path)
    assert report_path.is_file()
    assert test_gates._read_playwright_report(report_path) == report


def test_service_report_path_is_unique_and_never_falls_back_to_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each service run needs its own complete report file."""
    monkeypatch.setattr(test_gates, "ROOT", tmp_path)
    report = {
        "suites": [
            {
                "specs": [
                    {"tests": [{"results": [{"status": "passed"}]}]},
                    {"tests": [{"results": [{"status": "passed"}]}]},
                ]
            }
        ]
    }
    mode = "valid"
    report_paths: list[Path] = []

    def fake_run(command, *, label, log, env=None, **kwargs):
        del command, label, kwargs
        assert env is not None
        path = Path(env["PLAYWRIGHT_JSON_OUTPUT_FILE"])
        report_paths.append(path)
        # A valid-looking root object on stdout must never be used as a fallback.
        log.append(json.dumps(report) + "\n")
        if mode == "valid":
            path.write_text(json.dumps(report), encoding="utf-8")
        elif mode == "malformed":
            path.write_text("not-json", encoding="utf-8")
        return 0

    monkeypatch.setattr(test_gates, "_run", fake_run)

    assert test_gates.run_service(Namespace(pnpm="pnpm", expected=2)) == 0
    mode = "missing"
    assert test_gates.run_service(Namespace(pnpm="pnpm", expected=2)) == 4
    assert report_paths[1] != report_paths[0]
    assert not report_paths[1].exists()
    mode = "malformed"
    assert test_gates.run_service(Namespace(pnpm="pnpm", expected=2)) == 4
    assert report_paths[2] != report_paths[1]
    assert report_paths[2].is_file()


def test_native_gate_rejects_wrapper_only_flash_next_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four green focused tests must not replace the original Flash-Next body."""
    names = (
        "tyloinfer-numeric-assertions-test",
        "tyloinfer-cuda-numeric-assertions-test",
        "tyloinfer-cuda-attention-prefix-test",
        "tyloinfer-cuda-tensor-capture-lifetime-test",
    )
    monkeypatch.setattr(test_gates, "ROOT", tmp_path)
    monkeypatch.setattr(test_gates, "_run", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        test_gates, "_run_strict_ctest", lambda *args: (0, 4, 4, False, names)
    )
    result = test_gates.run_native(Namespace(
        build_dir="build", target=None, require_resource=[], cmake="cmake",
        ctest="ctest", config="Release", regex=test_gates.DEFAULT_NATIVE_REGEX,
        required_test=None, expected=None,
    ))
    assert result != 0


@pytest.mark.parametrize(
    ("communication", "missing"),
    [(False, None), (False, "mfq-model-source-test"),
     (True, None), (True, "mfq-request-executor-test")],
)
def test_cpu_profile_rejects_unregistered_required_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, communication: bool, missing: str | None
) -> None:
    """A green discovered subset must not hide an unregistered important test."""
    names = {
        "tyloinfer-numeric-assertions-test", "mfq-native-tensor-test",
        "mfq-model-source-test", "mfq-file-range-reader-test",
        "mfq-prefill-budget-test", "mfq-speculative-commit-test",
    }
    if communication:
        names.update({
            "mfq-tokenizer-test", "mfq-tokenizer-identity-test", "mfq-text-processor-test",
            "mfq-inference-test", "mfq-request-executor-test", "mfq-shared-execution-test",
            "mfq-speculative-sequence-test", "tyloinfer-session-cache-restore-test",
            "mfq-paged-prefix-cache-test", "mfq-decision-test",
            "mfq-runtime-transport-test", "mfq-runtime-profile-test",
        })
    if missing:
        names.remove(missing)
    monkeypatch.setattr(test_gates, "ROOT", tmp_path)
    monkeypatch.delenv("TYLOINFER_CPU_BUILD_DIR", raising=False)
    monkeypatch.delenv("TYLOINFER_CPU_COMM_BUILD_DIR", raising=False)
    monkeypatch.setattr(test_gates, "_run", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        test_gates, "_run_strict_ctest",
        lambda *args: (0, len(names), len(names), False, tuple(names)),
    )
    code, record = test_gates._run_cpu_native([], communication=communication)
    assert (code == 0) is (missing is None)
    assert record["missing_tests"] == ([missing] if missing else [])


def test_plugin_gate_rejects_skips_even_when_node_exits_successfully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An opt-out must not count as a successful Host/Client validation."""
    for package, name in (("host", "model-actions"), ("studio", "model-management")):
        tests = tmp_path / "plugins" / package / "tests"
        tests.mkdir(parents=True)
        (tests / f"{name}.test.mjs").write_text("", encoding="utf-8")
    monkeypatch.setattr(test_gates, "ROOT", tmp_path)

    def skipped_report(command, *, log, **kwargs):
        if "--test" in command:
            log.append("# tests 2\n# pass 1\n# fail 0\n# cancelled 0\n# skipped 1\n# todo 0\n")
        return 0

    monkeypatch.setattr(test_gates, "_run", skipped_report)
    code, records = test_gates._run_plugins([])
    assert code != 0
    assert all(record["status"] == "failed" for record in records.values())
