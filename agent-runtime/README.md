# agent-runtime

Dependency slot for the DeepSeek Harness (DSH) sidecar. This directory is
not a service and contains no code: only the npm manifest that installs
`@deepseek-ai/dsh`, which the **Python backend** launches and drives.

- Consumer: `tyloinfer/server/services/agent_runtime.py` locates
  `node_modules/@deepseek-ai/dsh/lib/bin.js` (and a bundled `node`
  executable) here at runtime; `tyloinfer/server/api/routes/agent.py`
  exposes it under `/api/agent/*`.
- Packaging: `packaging/stage_agent_runtime.py` copies this manifest,
  runs `npm ci`, and bundles the Node binary into releases;
  `packaging/build_portable_windows.py` ships the staged directory.
- Requirement: Node.js >= 24.12 (DSH engine constraint).
- The native C++ server does not serve `/api/agent/*`; Studio's agent UI
  is gated off until that wiring is verified.

To develop against the Python backend's agent feature, run `npm ci` here
first, otherwise startup reports the bundled runtime as not installed.
