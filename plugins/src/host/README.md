# TyloInfer Host integration

This local development bundle composes the unmodified official DSH provider
adapter with a TyloInfer route and a protocol-status plugin. It targets the
verified DSH `0.2.2-rc.2` composition; it is not a Core fork or a completed
redistributable release.

The default provider uses the local inference API and starts with an empty
model list. Configure/discover actual models through the official Models
settings; installing the bundle does not load weights or run inference. Other
official provider routes and their credential services remain available.

The same package ships the browser Client from [`../client/`](../client/).
DSH's Client module system loads it rather than mounting another application root.

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
than as success or cancellation. See [model bridge RPC](../../../docs/api/agent-model-bridge.md)
for payloads and the current deployment/verification limits. No plugin load,
page refresh or inventory read initiates a model load.

`GET /tyloinfer/status` reports the configured local route's health/protocol,
model-count configuration, native settings namespace names, and Loader entry
activation. It returns no credentials, profile paths, or session contents.
The optional `connectionFile` Config field writes a private authenticated URL
for local validation; never publish that file or log its contents.

Agent model/tool execution migration, automatic model-capability discovery,
independent-upgrade and final packaging checks, full compatibility tests, and
release artwork are not completed by this bundle. The legacy SDK prototype remains separate until
those migrations are validated.
