import json
import os
import shutil
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_payload import agent_payload, build_agent_payload


class AgentPayloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tylo-agent-payload-test-")
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.ta = self.project / "app/src-tauri"
        self.ta.mkdir(parents=True)
        self.config = {
            "bundle": {
                "resources": {
                    "../web/dist": "web-ui",
                    "../../packaging/windows-runtime/": "",
                    "../../packaging/Resources/tyloinfer-server": "tyloinfer-server",
                    "../../packaging/Resources/licenses": "licenses",
                },
                "externalBin": ["../../packaging/sidecars/mfq-runtime"],
            }
        }
        for name in (
            "app/src-tauri/agent/runtime/node.exe",
            "app/src-tauri/agent/runtime/boot.mjs",
            "app/src-tauri/agent/runtime/node_modules/@tyloinfer/dsh-plugin/package.json",
            "packaging/windows-runtime/vcruntime140.dll",
            "packaging/Resources/tyloinfer-server/tyloinfer-server.exe",
            "packaging/Resources/licenses/NOTICE",
            "packaging/sidecars/mfq-runtime-x86_64-pc-windows-msvc.exe",
        ):
            path = self.project / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("fixture: " + name).encode())
        self.save_config()
        self.archive = self.project / "packaging/build/agent-payload.zip"

    def save_config(self):
        (self.ta / "tauri.release-windows.conf.json").write_text(json.dumps(self.config))

    def test_archive_preserves_installed_layout_and_all_source_bytes(self):
        expected = agent_payload(self.project)
        build_agent_payload(self.project, self.archive)
        with zipfile.ZipFile(self.archive) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(set(archive.namelist()), set(expected))
            for name, source in expected.items():
                self.assertEqual(archive.read(name), source.read_bytes())
            self.assertNotIn("web-ui/index.html", archive.namelist())
            self.assertEqual(archive.getinfo("agent-runtime/node.exe").compress_type, zipfile.ZIP_DEFLATED)

    def test_missing_required_runtime_fails_before_replacing_an_archive(self):
        self.archive.parent.mkdir(parents=True)
        self.archive.write_bytes(b"existing release")
        (self.ta / "agent/runtime/node.exe").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "agent-runtime/node.exe"):
            build_agent_payload(self.project, self.archive)
        self.assertEqual(self.archive.read_bytes(), b"existing release")

    def test_unsafe_archive_paths_are_rejected(self):
        for destination in ("../escape", "C:/escape", "/escape", "bad\ncommand"):
            with self.subTest(destination=destination):
                self.config["bundle"]["resources"]["../../packaging/windows-runtime/"] = destination
                self.save_config()
                with self.assertRaisesRegex(ValueError, "Unsafe Agent archive path"):
                    agent_payload(self.project)

    def test_case_insensitive_destination_collisions_are_rejected(self):
        self.config["bundle"]["resources"]["../../packaging/windows-runtime/vcruntime140.dll"] = "VCRUNTIME140.DLL"
        self.save_config()
        with self.assertRaisesRegex(ValueError, "Duplicate Agent archive path"):
            agent_payload(self.project)

    def test_failed_compression_keeps_previous_archive_and_removes_partial_file(self):
        self.archive.parent.mkdir(parents=True)
        self.archive.write_bytes(b"existing release")
        with patch.object(zipfile.ZipFile, "write", side_effect=OSError("disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                build_agent_payload(self.project, self.archive)
        self.assertEqual(self.archive.read_bytes(), b"existing release")
        self.assertFalse(self.archive.with_suffix(".zip.partial").exists())

    def test_installer_reserves_expanded_payload_disk_space(self):
        (self.ta / "agent/runtime/node.exe").write_bytes(b"runtime" * 500)
        expected_kib = (sum(path.stat().st_size for path in agent_payload(self.project).values()) + 1023) // 1024
        build_agent_payload(self.project, self.archive)
        manifest = self.archive.with_name("agent-payload-uninstall.nsh").read_text()
        self.assertIn(f"!define TYLOINFER_AGENT_PAYLOAD_KIB {expected_kib}\n", manifest)

    def test_uninstaller_targets_only_payload_paths_and_never_profile_data(self):
        build_agent_payload(self.project, self.archive)
        script = self.archive.with_name("agent-payload-uninstall.nsh").read_text()
        self.assertIn('RMDir /r "$INSTDIR\\agent-runtime"', script)
        self.assertIn('RMDir /r "$INSTDIR\\tyloinfer-server"', script)
        self.assertIn('Delete "$INSTDIR\\mfq-runtime.exe"', script)
        self.assertIn('Delete "$INSTDIR\\vcruntime140.dll"', script)
        self.assertNotIn('RMDir /r "$INSTDIR"', script)
        self.assertNotIn("APPDATA", script)

    @unittest.skipUnless(os.name == "nt" and shutil.which("makensis"),
                         "Windows NSIS toolchain is not on PATH")
    def test_windows_installer_extracts_and_uninstalls_payload_with_spaces(self):
        build_agent_payload(self.project, self.archive)
        hooks = self.project / "packaging/windows_agent_hooks.nsh"
        hooks.write_text((Path(__file__).resolve().parents[1] / "windows_agent_hooks.nsh").read_text())
        install = self.project / "installed Agent with spaces"
        installer = self.project / "installer-test.exe"
        script = self.project / "installer-test.nsi"
        script.write_text(f'''Unicode true
Name "Agent payload hook regression"
OutFile "{installer}"
InstallDir "{install}"
RequestExecutionLevel user
SilentInstall silent
!include "{hooks}"
Section
  !insertmacro NSIS_HOOK_PREINSTALL
  SetOutPath "$INSTDIR"
  File "/oname=agent-payload.zip" "{self.archive}"
  !insertmacro NSIS_HOOK_POSTINSTALL
  WriteUninstaller "$INSTDIR\\uninstall.exe"
SectionEnd
Section "Uninstall"
  !insertmacro NSIS_HOOK_PREUNINSTALL
SectionEnd
''', encoding="utf-8")
        compiled = subprocess.run([shutil.which("makensis"), "-INPUTCHARSET", "UTF8", "-V2", str(script)],
                                  text=True, capture_output=True, timeout=60)
        self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
        subprocess.run([str(installer), "/S"], check=True, timeout=30)
        for name, source in agent_payload(self.project).items():
            self.assertEqual((install / name).read_bytes(), source.read_bytes())
        self.assertFalse((install / "agent-payload.zip").exists())
        sentinel = install / "user-file.txt"
        sentinel.write_text("keep user data")
        # NSIS requires its final _?= path without quotes, even when it has spaces.
        command = subprocess.list2cmdline([str(install / "uninstall.exe"), "/S"]) + " _?=" + str(install)
        subprocess.run(command, check=True, timeout=30)
        for name in agent_payload(self.project):
            self.assertFalse((install / name).exists(), name)
        self.assertEqual(sentinel.read_text(), "keep user data")


if __name__ == "__main__":
    unittest.main()
