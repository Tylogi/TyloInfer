# Agent integration

The TyloInfer Agent product runs the complete official DSH Host with external TyloInfer Host and Client plugins. Use the independent [Host launcher](../agent-host.md); native sessions, tasks, settings, providers, permissions and credentials belong to DSH. The inference core remains a separate HTTP service and does not require the Python DSH SDK.

## Retired Python facade

The earlier `/api/v1/agent` task and display-settings facade is no longer mounted by the inference core. Its source was moved outside the inference package to a non-distributed compatibility-test directory under `plugins/tyloinfer-host/legacy-python/`. It is not an alternative shipping Agent implementation and is excluded from the Host bundle's package file list.

The former `agent` Python extra and instructions to start a second SDK settings carrier do not apply to the supported product. Core health, inference, model management, jobs, sessions and standalone Studio APIs remain independent of DSH. Native Agent access uses the authenticated Host entry, not the core origin.

No existing Agent database, settings profile, credential or native Host session is moved, deleted or automatically converted by this change. Source relocation is not a data migration. Earlier facade regression tests opt in to its compatibility factory; they are not evidence that the core still publishes those routes.

## Standalone Studio

TyloInfer Studio remains a pure inference desktop application and WebUI. It must not import DSH or require the Agent UI build. Its own display preferences remain in the Studio settings layer; the Agent product uses DSH's native settings service. A removed SDK facade must not be reintroduced solely to make independent Studio settings work.

## Verification boundaries

The supported integration contract is split into [model bridge](agent-model-bridge.md), [inference page reads](agent-inference-pages.md), [maintenance](agent-maintenance-bridge.md), and [model/download actions](agent-model-actions.md). Each document distinguishes implementation, deployment, isolated API checks and actual user-operation evidence. A successful bridge test is not an end-to-end real-model Agent acceptance result.

This migration is pending the corresponding isolated validation and the next coordinated core deployment window. Do not restart an existing core or Host just to apply documentation changes. DS Bot-specific collaboration and native desktop distribution remain separate incomplete work.
