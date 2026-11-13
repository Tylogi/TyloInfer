"""Check documentation links, examples and CLI/API contracts without modifying files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tyloinfer.cli import _build_parser  # noqa: E402


def check() -> None:
    files = [ROOT / "README.md", ROOT / "README.zh-CN.md", *sorted((ROOT / "docs").rglob("*.md"))]
    errors: list[str] = []
    json_count = 0
    link_count = 0
    for file in files:
        text = file.read_text(encoding="utf-8")
        relative = file.relative_to(ROOT)
        if re.search(r"\bmfq\s+(?:build|serve)\b|\bmfq-minicpmo-realtime\b", text):
            errors.append(f"{relative}: stale public CLI command")
        for language, block in re.findall(r"```(json|jsonl)\s*\n(.*?)```", text, re.S):
            for value in block.splitlines() if language == "jsonl" else [block]:
                if not value.strip():
                    continue
                try:
                    json.loads(value)
                    json_count += 1
                except json.JSONDecodeError as error:
                    errors.append(f"{relative}: invalid {language} example: {error}")
        for raw in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            target = raw.strip().strip("<>")
            url = urlsplit(target)
            if url.scheme or url.netloc:
                continue
            destination = (file.parent / unquote(url.path)).resolve() if url.path else file
            link_count += 1
            if not destination.exists():
                errors.append(f"{relative}: missing local link {raw}")
            elif url.fragment and destination.suffix == ".md":
                body = destination.read_text(encoding="utf-8")
                anchors = set(re.findall(r'<a\s+(?:id|name)="([^"]+)"', body))
                for heading in re.findall(r"^#{1,6}\s+(.+)$", body, re.M):
                    heading = re.sub(r"[`*]", "", heading).lower()
                    anchors.add(re.sub(r"[^\w\- ]", "", heading).replace(" ", "-"))
                if unquote(url.fragment) not in anchors:
                    errors.append(f"{relative}: missing Markdown anchor {raw}")

    schema = json.loads((ROOT / "tyloinfer/server/protocol/openapi.json").read_text())
    methods = {"get", "post", "put", "patch", "delete", "options", "head", "trace"}
    operations = {(m.upper(), p): op for p, item in schema["paths"].items() for m, op in item.items() if m in methods}
    api = (ROOT / "docs/api/http.md").read_text()
    rows = re.findall(r"\| `(GET|POST|PUT|PATCH|DELETE)` \| `(/api/[^`]+)` \| ([^\n]+)", api)
    documented = {(method, path) for method, path, _ in rows}
    if documented != set(operations):
        errors.append(f"API inventory differs: missing={sorted(set(operations)-documented)}, stale={sorted(documented-set(operations))}")
    # Verify named body/response schemas for each table entry too.
    for method, path, rest in rows:
        operation = operations.get((method, path), {})
        for name in re.findall(r"`(?:\d{3} )?([A-Z][A-Za-z0-9]+)`", rest):
            if name == "Object":
                continue
            if name not in schema["components"]["schemas"] or f"/schemas/{name}" not in json.dumps(operation):
                errors.append(f"API {method} {path}: schema {name} differs from contract")

    parser = _build_parser()
    subcommands = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    if "mfq" in subcommands.choices or parser.prog != "tyloinfer":
        errors.append("Unexpected public CLI identity")
    serve = subcommands.choices["serve"]
    options = {flag for action in serve._actions for flag in action.option_strings}
    guide = (ROOT / "docs/cli/serve.md").read_text()
    for flag in ("--dflash2-draft", "--mtp-native-draft", "--spec-min-p", "--mtp-min-probability", "--draft-conf", "--dflash2-draft-confidence", "--mtp-cost-aware", "--no-mtp-cost-aware"):
        if flag not in options or flag not in guide:
            errors.append(f"Speculation CLI/doc mismatch: {flag}")
    if f"`./{serve.get_default('data_dir')}`" not in guide:
        errors.append("Documented data directory differs from CLI default")
    if "must point to an existing `.mfq`" in guide:
        errors.append("Obsolete model-format restriction")

    if errors:
        raise SystemExit("FAIL:\n" + "\n".join(errors))
    print(f"PASS: {len(files)} Markdown files; {json_count} JSON examples; {link_count} local links/anchors")
    print(f"PASS: API table matches {len(schema['paths'])} paths / {len(operations)} operations and named schemas")
    print("PASS: public CLI identity, data-dir default and speculative decoding options")


if __name__ == "__main__":
    check()
