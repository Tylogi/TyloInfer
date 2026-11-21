# TyloInfer Host integration

This local development bundle composes the unmodified official DSH provider
adapter with a TyloInfer route and a protocol-status plugin. It targets the
verified DSH `0.2.2-rc.2` composition; it is not a Core fork or a completed
redistributable release.

The default provider starts with an empty model list and connects to the local
inference API. The Host periodically registers loaded model IDs or their
serving aliases, preserving settings for matching IDs. Select a registered
model and configure its context and output budgets in DSH. See the
[model registration and output budget guide](../../docs/agent-host.md#模型登记和输出预算).
Other official provider routes and their credential services remain available.

The product Client package is separate. Its browser build must be loaded by the
official Client module system rather than mounted as another application root.

Core inventory reads use the independent Connection RPC channel `/tyloinfer`,
endpoint `core/read`, with a fixed resource allowlist. The official Typert
Gateway exclusively owns the shared `/api` interceptor; product code must not
register another one there. The Web composition supplies Connection's explicit
`webStartup` and `webServer` dependencies for its independent HTTP carrier.
Native settings, permissions and Session RPCs remain official Gateway methods.

The model-management bridge source adds `loaded_models`, explicit load/unload,
and model-job get/cancel operations on the same independent product channel.
It preserves native operation tickets and states; writes are never retried,
and a submitted request with an uncertain result is reported as unknown rather
than as success or cancellation. See [model bridge RPC](../../docs/api/agent-model-bridge.md)
for payloads and the current deployment/verification limits. No plugin load,
page refresh or inventory read initiates a model load.

`GET /tyloinfer/status` reports the configured local route's health/protocol,
model-count configuration, native settings namespace names, and Loader entry
activation. It returns no credentials, profile paths, or session contents.
The optional `connectionFile` Config field writes a private authenticated URL
for local validation; never publish that file or log its contents.

Web chat against a loaded TyloInfer model was verified with the pinned DSH
release in both the standalone Agent and DS Bot 0.3.8. The
[launch guide](../../docs/agent-host.md) covers both installations. The legacy
SDK prototype stays in the compatibility-test directory.
