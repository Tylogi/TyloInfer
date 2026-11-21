# Legacy Python facade compatibility sources

These sources were removed from the inference package and its default API composition. They preserve the earlier SDK facade for isolated compatibility regression tests only; they are not the TyloInfer Agent product, not a second shipped Agent runtime, and not included in the Host bundle's `files` list.

The supported Agent entry uses the complete official DSH Host and the external TyloInfer plugins. Its sessions, settings, permissions, providers and credentials use the official services. The inference core neither imports this directory nor exposes `/api/v1/agent` by default. The core package no longer declares the old Python Agent SDK extra.

Do not add this directory to a production `PYTHONPATH`, mount its routes into a shipping core, or start its separate settings carrier as a substitute for the complete Host. Compatibility tests opt in through `studio/tests/legacy_agent_support.py`; tests requiring the earlier SDK process remain separately opt-in.

No existing task database, settings profile, native Host session or credential was moved or deleted. Legacy data is not converted into native DSH sessions automatically. This source relocation does not authorize reading private state or changing any running deployment.
