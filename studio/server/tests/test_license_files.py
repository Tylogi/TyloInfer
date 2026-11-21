from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[3]
PACKAGE_ROOT = PROJECT_ROOT / "studio" / "server"


def test_package_license_copies_match_repo_root():
    # setuptools resolves license-files relative to studio/server, so the texts
    # are duplicated there. This keeps the two copies from drifting apart.
    for name in ("LICENSE", "NOTICE"):
        assert (PACKAGE_ROOT / name).read_bytes() == (PROJECT_ROOT / name).read_bytes()

    names = sorted(p.name for p in (PROJECT_ROOT / "LICENSES").glob("*.txt"))
    copied = sorted(p.name for p in (PACKAGE_ROOT / "LICENSES").glob("*.txt"))
    assert names == copied
    for name in names:
        left = (PROJECT_ROOT / "LICENSES" / name).read_bytes()
        right = (PACKAGE_ROOT / "LICENSES" / name).read_bytes()
        assert left == right


def test_every_license_text_is_indexed_in_notice():
    notice = (PACKAGE_ROOT / "NOTICE").read_text(encoding="utf-8")
    for path in sorted((PACKAGE_ROOT / "LICENSES").glob("*.txt")):
        assert path.name in notice, f"NOTICE does not reference {path.name}"
