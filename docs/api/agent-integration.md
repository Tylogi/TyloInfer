# Agent integration

The supported Agent product uses the complete official DSH Host plus external TyloInfer Host and Client plugins. Start it through the independent [Host launcher](../agent-host.md). Native sessions, settings, permissions, providers and credentials remain owned by DSH.

The Python inference core remains a separate HTTP service and does not require the Python DSH SDK. Legacy SDK task and settings carriers are not part of this product. Existing user data is not automatically converted into native DSH sessions.

Standalone Studio remains an independent inference desktop application and WebUI. Its own display settings must not depend on DSH, and its build must not require the Agent Client build.

The supported fixed operations are described in [model management](agent-model-bridge.md), [page reads](agent-inference-pages.md), [maintenance](agent-maintenance-bridge.md), [model and download actions](agent-model-actions.md), and [restricted configuration](agent-product-boundaries.md). Only explicit user actions may submit writes; inventory reads and reconnects never load weights or replay a write.

Current-version isolated API and lifecycle checks are not evidence for real-weight tools, arbitrary cross-release upgrades, DS Bot collaboration or final desktop distribution. Those remain separate acceptance work.
