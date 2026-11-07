"""Build the MFQ Server OpenAPI contract in memory from the contract app."""

from __future__ import annotations

from typing import Any

from tyloinfer.server.api import create_contract_app
from tyloinfer.server.protocol.models import PROTOCOL_VERSION

REALTIME_EVENTS = [
    "input_audio.delta",
    "input_audio.commit",
    "response.text.delta",
    "response.reasoning.delta",
    "response.tool_call.delta",
    "response.audio.delta",
    "response.interrupted",
    "response.completed",
    "session.state",
    "runtime.metrics",
    "error",
]


def build_openapi_schema() -> dict[str, Any]:
    schema: dict[str, Any] = create_contract_app().openapi()
    schema["x-mfq-protocol-version"] = PROTOCOL_VERSION
    schema["x-mfq-websocket"] = {
        "path": "/api/v1/realtime",
        "frameSchema": "#/components/schemas/RealtimeFrame",
        "events": REALTIME_EVENTS,
    }
    from tyloinfer.server.protocol.models import RealtimeFrame

    realtime_schema = RealtimeFrame.model_json_schema(ref_template="#/components/schemas/{model}")
    nested_schemas = realtime_schema.pop("$defs", {})
    schema["components"]["schemas"].update(nested_schemas)
    schema["components"]["schemas"]["RealtimeFrame"] = realtime_schema
    return schema
