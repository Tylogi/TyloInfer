"""Shared Agent workbench routes for Web and desktop clients."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from server.api import create_app as create_core_app
from server.services.service import ServiceError

from .service import AgentService


@asynccontextmanager
async def legacy_lifespan(app):
    try:
        yield
    finally:
        if app.state.agent is not None:
            await app.state.agent.aclose()


router = APIRouter(prefix="/api/v1/agent", tags=["legacy-agent"], lifespan=legacy_lifespan)


def create_app(*args, **kwargs):
    app = create_core_app(*args, **kwargs)
    app.state.agent = None
    app.include_router(router)
    return app


class CreateAgentSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(default="New task", min_length=1, max_length=120)
    model: str = Field(min_length=1, max_length=255)

    @field_validator("title", "model")
    @classmethod
    def reject_blank(cls, value):
        if not value.strip():
            raise ValueError("value must not be blank")
        return value


class CreateAgentRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: str = Field(min_length=1, max_length=16384)
    request_id: UUID

    @field_validator("input")
    @classmethod
    def reject_blank(cls, value):
        if not value.strip():
            raise ValueError("input must not be blank")
        return value


class UpdateAgentSettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["update", "replace"] = "update"
    values: dict[str, JsonValue] = Field(max_length=16)
    expected_revision: int = Field(ge=0, le=9007199254740991, strict=True)
    generation: UUID

    @field_validator("values")
    @classmethod
    def validate_size(cls, value):
        if len(json.dumps(value, allow_nan=False).encode("utf-8")) > 32768:
            raise ValueError("settings values exceed the request limit")
        return value


async def require_agent(request: Request) -> AgentService:
    agent = request.app.state.agent
    if agent is None:
        service = request.app.state.service
        if service is None:
            raise ServiceError(501, "contract_only", "Agent execution is not configured")
        agent = AgentService(service)
        request.app.state.agent = agent
    return agent


AgentDependency = Annotated[AgentService, Depends(require_agent)]


@router.get("/capabilities")
async def capabilities(agent: AgentDependency) -> dict:
    return agent.capabilities()


@router.get("/settings")
async def settings(response: Response, agent: AgentDependency) -> dict:
    response.headers["Cache-Control"] = "no-store"
    return await agent.settings.describe()


@router.patch("/settings/{namespace}")
async def update_settings(
    namespace: str, body: UpdateAgentSettingsRequest, response: Response, agent: AgentDependency
) -> dict:
    response.headers["Cache-Control"] = "no-store"
    return await agent.settings.update(
        namespace,
        mode=body.mode,
        values=body.values,
        expected_revision=body.expected_revision,
        generation=str(body.generation),
    )


@router.get("/sessions")
async def sessions(agent: AgentDependency) -> dict:
    return {"data": agent.store.sessions()}


@router.post("/sessions", status_code=201)
async def create_session(body: CreateAgentSessionRequest, agent: AgentDependency) -> dict:
    return agent.store.create_session(body.title, body.model)


@router.get("/sessions/{session_id}")
async def session(session_id: UUID, agent: AgentDependency) -> dict:
    return agent.store.session(str(session_id), runs=True)


@router.post("/sessions/{session_id}/runs", status_code=202)
async def create_run(session_id: UUID, body: CreateAgentRunRequest, agent: AgentDependency) -> dict:
    return agent.start(str(session_id), str(body.request_id), body.input)


@router.get("/runs/{run_id}")
async def run(run_id: UUID, agent: AgentDependency) -> dict:
    return agent.store.run(str(run_id))


@router.post("/runs/{run_id}/cancel")
async def cancel(run_id: UUID, agent: AgentDependency) -> dict:
    return await agent.cancel(str(run_id))


@router.get("/runs/{run_id}/events")
async def events(
    run_id: UUID, request: Request, agent: AgentDependency, after: int = Query(default=0, ge=0)
) -> StreamingResponse:
    run = agent.store.run(str(run_id))
    cursor = request.headers.get("last-event-id")
    if cursor is not None:
        try:
            if len(cursor) > 20 or int(cursor) < 0:
                raise ValueError("invalid cursor")
            after = max(after, int(cursor))
        except ValueError as error:
            raise ServiceError(
                400, "invalid_event_cursor", "Event cursor must be an integer"
            ) from error
    if after < 0 or after > run["last_event_id"]:
        raise ServiceError(400, "invalid_event_cursor", "Event cursor is outside the run history")

    async def stream():
        async for event in agent.events(str(run_id), after):
            if await request.is_disconnected():
                return
            if event is None:
                yield ": keep-alive\n\n"
            else:
                yield f"id: {event['event_id']}\nevent: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
