# DSH vendor boundary

`upstream/` contains the complete, unmodified upstream source checkout pinned by
`UPSTREAM.json`. Product changes belong in `plugins/`.

The source is pinned to the official `dsh-v0.2.2-rc.2` release tag, matching the
validated official runtime package version `0.2.2-rc.2`. No local Core patches
are applied. This is release-version alignment, not a verified reproducible
build: the runtime uses published official packages, not a local source build.
Install the runtime dependencies from `runtime/package-lock.json` under `runtime/`.

TyloInfer Host and Client customizations belong in the sibling project
`plugins/` directory. They depend on public DSH services and the TyloInfer HTTP
protocol, not on copied internal services. Profiles and private authenticated
connection handoffs are runtime data, not vendor source or release assets.

The Python inference API has no DSH dependency. The Agent Host uses the
official base and Web bundles.

The product-owned `scripts/agent_host.py` prepares the complete default profile
and launches only the official profile CLI. It accepts explicit runtime, plugin
and private-state roots without importing the inference core.

See [Agent Host setup](../../docs/agent-host.md) for installation and startup.
