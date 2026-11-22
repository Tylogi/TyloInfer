# Agent integration

The TyloInfer Agent product runs the complete official DSH Host with the TyloInfer plugin bundle, which contains both Host and Client. Start it with the official `dsh` CLI as described in [Host startup](../agent-host.md); native sessions, tasks, settings, providers, permissions and credentials belong to DSH. The inference core remains a separate HTTP service and does not require the Python DSH SDK.

## Retired Python facade

The earlier `/api/v1/agent` task and display-settings facade is no longer mounted by the inference core, and its source is no longer kept in this repository. It is not an alternative shipping Agent implementation.

The former `agent` Python extra and instructions to start a second SDK settings carrier do not apply to the supported product. Core health, inference, model management, jobs, sessions and standalone Studio APIs remain independent of DSH. Native Agent access uses the authenticated Host entry, not the core origin.

No existing Agent database, settings profile, credential or native Host session is moved, deleted or automatically converted by this change. Deleting the source is not a data migration. Nothing in this repository now publishes, imports or tests that facade.

## Standalone Studio

TyloInfer Studio remains a pure inference desktop application and WebUI. It must not import DSH or require the Agent UI build. Its own display preferences remain in the Studio settings layer; the Agent product uses DSH's native settings service. A removed SDK facade must not be reintroduced solely to make independent Studio settings work.

## Verification boundaries

The supported integration contract is split into [model bridge](agent-model-bridge.md), [inference page reads](agent-inference-pages.md), [maintenance](agent-maintenance-bridge.md), and [model/download actions](agent-model-actions.md). Each document distinguishes implementation, deployment, isolated API checks and actual user-operation evidence. A successful bridge test is not an end-to-end real-model Agent acceptance result.

Web chat against a loaded TyloInfer model was verified in the standalone Agent and in DS Bot `0.3.8`, both on DSH `0.2.2-rc.2`. Installation and provider settings are documented in the [Host and DS Bot guide](../agent-host.md).
