"""Load the non-distributed legacy adapter only for its compatibility regression tests."""

from __future__ import annotations

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2] / "plugins" / "host" / "legacy-python"
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from tyloinfer_legacy_agent.routes import create_app  # noqa: E402
from tyloinfer_legacy_agent.service import AgentService, AgentStore  # noqa: E402

__all__ = ["AgentService", "AgentStore", "create_app"]
