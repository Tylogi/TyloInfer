"""Independent Agent launcher checks without building or running a model."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts import agent_host


def make_layout(root: Path):
    runtime, plugins = root / "runtime", root / "plugins"
    for name in agent_host.OFFICIAL:
        manifest = runtime / "node_modules" / name / "package.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"name": name, "version": agent_host.DSH_VERSION}), encoding="utf-8")
    for name, directory in agent_host.PRODUCTS.items():
        package = plugins / directory
        package.mkdir(parents=True)
        exports = {".": "./index.js"}
        if name.endswith("dsh-studio"):
            exports["./client"] = "./client.js"
        for target in exports.values():
            (package / target).write_text("", encoding="utf-8")
        (package / "package.json").write_text(json.dumps({
            "name": name, "engines": {"dsh": agent_host.DSH_VERSION}, "exports": exports,
        }), encoding="utf-8")
    carrier = runtime / "node_modules" / ".bin" / ("dsh.cmd" if os.name == "nt" else "dsh")
    carrier.parent.mkdir(parents=True)
    carrier.write_text("", encoding="utf-8")
    return runtime, plugins, carrier


def test_check_does_not_create_profile_or_start_process(tmp_path, monkeypatch, capsys):
    runtime, plugins, _ = make_layout(tmp_path)
    state = tmp_path / "private-state"
    monkeypatch.delenv("DSH_OFFICE_SIDECAR", raising=False)
    monkeypatch.setattr(agent_host.subprocess, "run", lambda *_a, **_k: pytest.fail("check must not launch a process"))
    assert agent_host.main(["--runtime-root", str(runtime), "--plugins-root", str(plugins), "--state-root", str(state), "--check"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["ready"] and output["starts_models"] is False
    assert output["bundles"] == agent_host.BUNDLES
    assert not state.exists()


@pytest.mark.parametrize("name", agent_host.OFFICIAL)
def test_official_package_versions_must_match(tmp_path, name):
    runtime, plugins, _ = make_layout(tmp_path)
    manifest = runtime / "node_modules" / name / "package.json"
    manifest.write_text(json.dumps({"name": name, "version": "0.2.2-rc.3"}), encoding="utf-8")
    with pytest.raises(agent_host.LauncherError, match="supported DSH"):
        agent_host.check_layout(runtime, plugins)


def test_missing_client_artifact_is_not_treated_as_ready(tmp_path):
    runtime, plugins, _ = make_layout(tmp_path)
    manifest = plugins / "tyloinfer-studio" / "package.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["exports"]["./client"] = "./not-built.js"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(agent_host.LauncherError, match="Missing built"):
        agent_host.check_layout(runtime, plugins)


def test_artifacts_cannot_escape_the_selected_plugin(tmp_path):
    runtime, plugins, _ = make_layout(tmp_path)
    manifest = plugins / "tyloinfer-studio" / "package.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["exports"]["./client"] = "./../tyloinfer-host/index.js"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(agent_host.LauncherError, match="Missing built"):
        agent_host.check_layout(runtime, plugins)


def test_isolated_environment_does_not_import_credentials_or_profile(tmp_path):
    inherited = {"PATH": "runtime-path", "DSH_HOME": "other-profile", "DSH_SECRET_DEMO": "not-retained",
                 "PROVIDER_API_KEY": "not-retained", "UNRELATED_APP_DATA": "not-retained",
                 "https_proxy": "not-retained", "DSH_PERMISSION_MODE": "danger-full-access", "NODE_OPTIONS": "unwanted"}
    environment = agent_host.isolated_environment(tmp_path, inherited)
    assert environment["PATH"] == inherited["PATH"]
    assert environment["DSH_HOME"] == str(tmp_path / "home")
    assert environment["USERPROFILE"] == str(tmp_path / "workspace")
    assert environment["TEMP"] == str(tmp_path / "home" / "tmp")
    assert not any(name in environment for name in ("DSH_SECRET_DEMO", "PROVIDER_API_KEY", "DSH_PERMISSION_MODE", "UNRELATED_APP_DATA", "https_proxy"))
    assert environment["NODE_OPTIONS"] == ""
    assert environment["DSH_TELEMETRY_DISABLED"] == "1"
    assert inherited["DSH_HOME"] == "other-profile"


def test_office_resources_are_refused(tmp_path):
    with pytest.raises(agent_host.LauncherError, match="Office"):
        agent_host.isolated_environment(tmp_path, {"DSH_OFFICE_SIDECAR": "active"})


def test_default_profile_retains_all_official_bundles_and_saved_settings(tmp_path):
    runtime, plugins, carrier = make_layout(tmp_path)
    assert agent_host.check_layout(runtime, plugins) == carrier
    state = tmp_path / "private-state"
    agent_host.prepare_profile(state, plugins, "http://127.0.0.1:8090/v1")
    profile = state / "home" / "profiles" / agent_host.PROFILE
    manifest = profile / "package.json"
    patch = profile / "cordis.patch.yml"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    assert value["dsh"]["profile"]["bundles"] == agent_host.BUNDLES
    assert set(value["dependencies"]) == set(agent_host.PRODUCTS)
    for name, directory in agent_host.PRODUCTS.items():
        assert (profile / "node_modules" / name).resolve() == (plugins / directory).resolve()
    entries = json.loads(patch.read_text(encoding="utf-8"))
    provider = entries[0]["config"]["providers"]["tyloinfer"]
    assert provider["models"] == []
    assert all(not row.get("disabled") for row in entries)
    entries.append({"id": "user-preference", "config": {"retained": True}})
    saved = json.dumps(entries)
    patch.write_text(saved, encoding="utf-8")
    original_manifest = manifest.read_bytes()
    agent_host.prepare_profile(state, plugins, "https://different.invalid/v1")
    assert patch.read_text(encoding="utf-8") == saved
    assert manifest.read_bytes() == original_manifest


@pytest.mark.parametrize("url", ["file:///model", "https://name:password@localhost/v1", "http://localhost/v1?key=value"])
def test_initial_endpoint_refuses_non_http_and_embedded_credentials(tmp_path, url):
    with pytest.raises(agent_host.LauncherError, match="endpoint"):
        agent_host.prepare_profile(tmp_path / "state", tmp_path / "plugins", url)
    assert not (tmp_path / "state").exists()


def test_process_command_is_only_official_loopback_profile_entry(tmp_path):
    carrier = tmp_path / "runtime" / "dsh"
    command = agent_host.launch_command(carrier, 8123)
    assert command == [str(carrier), "--profile", "tyloinfer-web", "--host", "127.0.0.1", "--port", "8123", "--no-open"]
    with pytest.raises(agent_host.LauncherError, match="port"):
        agent_host.launch_command(carrier, 65536)


def test_selected_installation_and_state_paths_can_contain_spaces(tmp_path):
    runtime, plugins, carrier = make_layout(tmp_path / "another installation")
    assert agent_host.check_layout(runtime, plugins) == carrier
    state = tmp_path / "another private state"
    agent_host.prepare_profile(state, plugins, "http://localhost:8120/v1")
    link = state / "home" / "profiles" / agent_host.PROFILE / "node_modules" / "@tyloinfer" / "dsh-host"
    assert link.resolve() == (plugins / "tyloinfer-host").resolve()


def test_existing_product_directory_is_never_overwritten(tmp_path):
    _runtime, plugins, _carrier = make_layout(tmp_path)
    state = tmp_path / "state"
    existing = state / "home" / "profiles" / agent_host.PROFILE / "node_modules" / "@tyloinfer" / "dsh-host"
    existing.mkdir(parents=True)
    marker = existing / "retained.txt"
    marker.write_text("retained", encoding="utf-8")
    with pytest.raises(agent_host.LauncherError, match="different installation"):
        agent_host.prepare_profile(state, plugins, "http://localhost:8120/v1")
    assert marker.read_text(encoding="utf-8") == "retained"


def test_launch_uses_selected_official_carrier_and_preserves_exit_code(tmp_path, monkeypatch):
    runtime, plugins, carrier = make_layout(tmp_path)
    state = tmp_path / "state"
    calls = []
    monkeypatch.delenv("DSH_OFFICE_SIDECAR", raising=False)

    def run(command, **options):
        calls.append((command, options))
        return agent_host.subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(agent_host.subprocess, "run", run)
    assert agent_host.main(["--runtime-root", str(runtime), "--plugins-root", str(plugins), "--state-root", str(state), "--port", "8123"]) == 7
    assert len(calls) == 1
    command, options = calls[0]
    assert command == agent_host.launch_command(carrier, 8123)
    assert options["cwd"] == state / "workspace"
    assert options["env"]["DSH_HOME"] == str(state / "home")
    assert options["env"]["TYLOINFER_DSH_VERSION"] == agent_host.DSH_VERSION


def test_open_existing_uses_private_connection_without_launching_host(tmp_path, monkeypatch, capsys):
    state = tmp_path / "private-state"
    state.mkdir()
    private_url = "http://127.0.0.1:8123/connect?token=example-local-test-token"
    (state / "connection.json").write_text(json.dumps({"url": private_url, "origin": "http://127.0.0.1:8123"}), encoding="utf-8")
    opened = []
    monkeypatch.setattr(agent_host.webbrowser, "open", lambda url, **options: opened.append((url, options)) or True)
    monkeypatch.setattr(agent_host.subprocess, "run", lambda *_a, **_k: pytest.fail("must not start a Host"))
    assert agent_host.main(["--state-root", str(state), "--port", "8123", "--open-existing"]) == 0
    assert opened == [(private_url, {"new": 2})]
    output = capsys.readouterr()
    assert "AGENT_BROWSER_OPEN_REQUESTED" in output.out
    assert "example-local-test-token" not in output.out + output.err
    assert not (state / "home").exists()


@pytest.mark.parametrize("url", [
    "https://external.invalid/connect?token=example-token",
    "http://127.0.0.1:8124/connect?token=example-token",
    "http://user:password@127.0.0.1:8123/connect?token=example-token",
    "javascript:example-token",
    "http://127.0.0.1:8123/api/settings/mutate?token=example-token",
])
def test_open_existing_refuses_unexpected_origins_ports_or_paths(tmp_path, url):
    (tmp_path / "connection.json").write_text(json.dumps({"url": url, "origin": "http://127.0.0.1:8123"}), encoding="utf-8")
    with pytest.raises(agent_host.LauncherError, match="loopback"):
        agent_host.open_existing(tmp_path, 8123, opener=lambda *_a, **_k: pytest.fail("must not open"))


def test_selected_runtime_version_comes_from_declared_compatibility_not_launcher_constant(tmp_path, monkeypatch, capsys):
    runtime, plugins, _ = make_layout(tmp_path)
    fixture_version = "0.2.2-rc.3"
    for name in agent_host.OFFICIAL:
        path = runtime / "node_modules" / name / "package.json"
        path.write_text(json.dumps({"name": name, "version": fixture_version}), encoding="utf-8")
    for _name, directory in agent_host.PRODUCTS.items():
        path = plugins / directory / "package.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["engines"]["dsh"] = fixture_version
        path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.delenv("DSH_OFFICE_SIDECAR", raising=False)
    assert agent_host.main(["--runtime-root", str(runtime), "--plugins-root", str(plugins), "--state-root", str(tmp_path / "state"), "--check"]) == 0
    assert json.loads(capsys.readouterr().out)["dsh"] == fixture_version
    assert not (tmp_path / "state").exists()


def test_rejected_runtime_replacement_leaves_saved_profile_ready_for_rollback(tmp_path, monkeypatch, capsys):
    runtime, plugins, _ = make_layout(tmp_path / "original")
    state = tmp_path / "state"
    agent_host.prepare_profile(state, plugins, "http://core.invalid/v1")
    profile = state / "home" / "profiles" / agent_host.PROFILE
    patch = profile / "cordis.patch.yml"
    original = patch.read_bytes()
    replacement, _, _ = make_layout(tmp_path / "replacement")
    path = replacement / "node_modules" / agent_host.OFFICIAL[0] / "package.json"
    path.write_text(json.dumps({"name": agent_host.OFFICIAL[0], "version": "0.2.2-rc.3"}), encoding="utf-8")
    monkeypatch.delenv("DSH_OFFICE_SIDECAR", raising=False)
    monkeypatch.setattr(agent_host.subprocess, "run", lambda *_a, **_k: pytest.fail("preflight must not start a runtime"))
    assert agent_host.main(["--runtime-root", str(replacement), "--plugins-root", str(plugins), "--state-root", str(state), "--prepare-only"]) == 1
    assert patch.read_bytes() == original
    assert agent_host.main(["--runtime-root", str(runtime), "--plugins-root", str(plugins), "--state-root", str(state), "--prepare-only"]) == 0
    assert patch.read_bytes() == original
    capsys.readouterr()


@pytest.mark.parametrize("mode", ["--check", "--prepare-only"])
def test_existing_profile_bindings_refuse_a_different_plugin_installation_before_writes(tmp_path, monkeypatch, capsys, mode):
    runtime, original_plugins, _ = make_layout(tmp_path / "original")
    _, replacement_plugins, _ = make_layout(tmp_path / "replacement")
    state = tmp_path / "private-state"
    agent_host.prepare_profile(state, original_plugins, "http://core.invalid/v1")
    profile = state / "home" / "profiles" / agent_host.PROFILE
    retained = {name: (profile / name).read_bytes() for name in ("package.json", "cordis.patch.yml")}
    monkeypatch.delenv("DSH_OFFICE_SIDECAR", raising=False)
    monkeypatch.setattr(agent_host.subprocess, "run", lambda *_a, **_k: pytest.fail("binding refusal must not launch"))
    assert agent_host.main(["--runtime-root", str(runtime), "--plugins-root", str(replacement_plugins),
        "--state-root", str(state), mode]) == 1
    assert all((profile / name).read_bytes() == content for name, content in retained.items())
    for name, directory in agent_host.PRODUCTS.items():
        assert (profile / "node_modules" / name).resolve() == (original_plugins / directory).resolve()
    output = capsys.readouterr()
    assert "different installation" in output.err and '"ready": true' not in output.out


def test_profile_bindings_are_preflighted_before_creating_any_package_or_profile_file(tmp_path, monkeypatch):
    _, plugins, _ = make_layout(tmp_path / "installation")
    state = tmp_path / "private-state"
    profile = state / "home" / "profiles" / agent_host.PROFILE
    conflict = profile / "node_modules" / "@tyloinfer" / "dsh-studio"
    conflict.mkdir(parents=True)
    marker = conflict / "retained.txt"
    marker.write_text("retained", encoding="utf-8")
    monkeypatch.setattr(agent_host, "_link_directory", lambda *_a, **_k: pytest.fail("preflight must check all bindings first"))
    with pytest.raises(agent_host.LauncherError, match="different installation"):
        agent_host.prepare_profile(state, plugins, "http://core.invalid/v1")
    assert marker.read_text(encoding="utf-8") == "retained"
    assert not (profile / "node_modules" / "@tyloinfer" / "dsh-host").exists()
    assert not any((profile / name).exists() for name in ("package.json", "cordis.patch.yml"))
    assert not any((state / name).exists() for name in ("workspace", "home/cache", "home/tmp"))


def test_broken_profile_binding_is_refused_without_replacing_it(tmp_path):
    _, plugins, _ = make_layout(tmp_path / "installation")
    state = tmp_path / "private-state"
    link = state / "home" / "profiles" / agent_host.PROFILE / "node_modules" / "@tyloinfer" / "dsh-studio"
    foreign = tmp_path / "removed-installation"
    foreign.mkdir()
    agent_host._link_directory(link, foreign)
    assert foreign.resolve() == (tmp_path / "removed-installation").resolve()
    foreign.rmdir()
    assert agent_host.os.path.lexists(link)
    with pytest.raises(agent_host.LauncherError, match="different installation"):
        agent_host.prepare_profile(state, plugins, "http://core.invalid/v1")
    assert agent_host.os.path.lexists(link)
    assert not (state / "workspace").exists()


def test_matching_profile_bindings_check_is_read_only(tmp_path, monkeypatch, capsys):
    runtime, plugins, _ = make_layout(tmp_path)
    state = tmp_path / "private-state"
    agent_host.prepare_profile(state, plugins, "http://core.invalid/v1")
    profile = state / "home" / "profiles" / agent_host.PROFILE
    retained = {name: (profile / name).read_bytes() for name in ("package.json", "cordis.patch.yml")}
    monkeypatch.delenv("DSH_OFFICE_SIDECAR", raising=False)
    monkeypatch.setattr(agent_host, "_link_directory", lambda *_a, **_k: pytest.fail("check must not link or prepare"))
    monkeypatch.setattr(agent_host.subprocess, "run", lambda *_a, **_k: pytest.fail("check must not launch"))
    assert agent_host.main(["--runtime-root", str(runtime), "--plugins-root", str(plugins),
        "--state-root", str(state), "--check"]) == 0
    assert all((profile / name).read_bytes() == content for name, content in retained.items())
    assert json.loads(capsys.readouterr().out)["ready"] is True


def test_open_existing_reports_browser_refusal_without_private_url(tmp_path, capsys, monkeypatch):
    (tmp_path / "connection.json").write_text(json.dumps({
        "url": "http://127.0.0.1:8123/connect?token=example-token", "origin": "http://127.0.0.1:8123",
    }), encoding="utf-8")
    monkeypatch.setattr(agent_host.webbrowser, "open", lambda *_a, **_k: False)
    assert agent_host.main(["--state-root", str(tmp_path), "--port", "8123", "--open-existing"]) == 1
    output = capsys.readouterr()
    assert "example-token" not in output.out + output.err
    assert "did not accept" in output.err
