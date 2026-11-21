"""Desktop package layout and the Rust shell's declared wiring.

The TypeScript side of the desktop client is tested by the web suite
(``npm --prefix studio/web test``); nothing here reads ``.ts``/``.tsx``. The Rust
side stays as a content guard because the Tauri crate needs GTK/WebKit system
libraries, so ``cargo test`` does not run on every host. Move these checks into
``src-tauri`` when a Rust test job exists.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDIO = ROOT / "studio/web"
TAURI = STUDIO / "src-tauri"
RUST = (TAURI / "src" / "main.rs").read_text(encoding="utf-8")
BUILD = (TAURI / "build.rs").read_text(encoding="utf-8")
UPDATER = (TAURI / "src" / "updater.rs").read_text(encoding="utf-8")


def test_studio_uses_one_package_for_web_and_desktop_clients():
    config = json.loads((TAURI / "tauri.conf.json").read_text(encoding="utf-8"))
    release_config = json.loads(
        (TAURI / "tauri.release-macos.conf.json").read_text(encoding="utf-8")
    )
    package = json.loads((STUDIO / "package.json").read_text(encoding="utf-8"))
    assert not (STUDIO / "web").exists()
    assert not (STUDIO / "desktop").exists()
    assert package["name"] == "@tyloinfer/studio"
    assert package["scripts"]["tauri"] == "tauri"
    assert config["build"]["frontendDist"] == "../dist"
    assert config["build"]["beforeBuildCommand"] == "pnpm build"
    assert config["identifier"] == "com.tylogi.tyloinfer"
    assert "icons/icon.ico" in config["bundle"]["icon"]
    assert "icons/icon.icns" in config["bundle"]["icon"]
    assert "IconDir::new" in BUILD
    assert "IconFamily::new" in BUILD
    assert "media-src 'self' asset: data: blob:" in config["app"]["security"]["csp"]
    assert release_config["bundle"]["macOS"]["hardenedRuntime"] is False


def test_studio_macos_icons_have_platform_padding():
    assert "const MACOS_ICON_SCALE: f64 = 0.79;" in BUILD
    assert 'std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")' in BUILD
    assert "render(size, MACOS_ICON_SCALE)" in BUILD
    assert "IconImage::from_rgba_data(size, size, render(size, 1.0))" in BUILD
    for size in (32, 128, 256):
        assert f'write_png(&directory.join("{size}x{size}.png"), {size}, png_scale);' in BUILD


def test_studio_starts_the_unified_local_server_and_bundled_runtime():
    assert 'Some("tyloinfer-server")' in RUST
    assert "Command::new" in RUST
    assert "studio_start_local" in RUST
    assert '.arg("server.serve")' in RUST
    assert '.arg("--data-dir")' in RUST
    assert '.arg("--db")' not in RUST
    assert 'command.arg("--running-executable")' in RUST
    assert "TYLOINFER_MLX_METALLIB" in RUST
    assert "TYLOINFER_AVFOUNDATION_VIDEO_LIBRARY" in RUST


def test_studio_keeps_web_ui_available_on_the_server_listener():
    assert '.arg("--no-web-ui")' not in RUST
    for platform in ("macos", "windows"):
        config = json.loads(
            (TAURI / f"tauri.release-{platform}.conf.json").read_text(encoding="utf-8")
        )
        assert config["bundle"]["resources"]["../dist"] == "web-ui"


def test_studio_supports_local_and_remote_server_connections():
    for command in ("studio_configure", "studio_start_local", "studio_select_model_directory"):
        assert command in RUST


def test_studio_can_select_and_load_an_external_mfq_directory_in_local_mode():
    assert "rfd::AsyncFileDialog::new()" in RUST
    assert ".pick_folder()" in RUST
    assert ".set_directory(initial_directory)" in RUST
    assert "path: path.to_string()" in RUST


def test_studio_uses_native_confirmation_dialogs_for_destructive_actions():
    assert "fn studio_confirm(message: String) -> bool" in RUST
    assert "rfd::MessageButtons::YesNo" in RUST
    assert "studio_confirm," in RUST


def test_studio_checks_releases_and_keeps_verified_versions_for_rollback():
    for command in (
        "studio_update_status",
        "studio_update_download",
        "studio_update_install",
        "studio_update_delete",
        "studio_update_progress",
    ):
        assert command in RUST
    assert "api.github.com/repos/Tylogi/TyloInfer/releases" in UPDATER
    assert "release SHA-256 verification failed" in UPDATER
    assert "snapshot_current_bundle" in UPDATER
    assert "replace_macos_bundle" in UPDATER
    assert "com.tylogi.tyloinfer" in UPDATER
