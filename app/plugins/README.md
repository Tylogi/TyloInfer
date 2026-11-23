# TyloInfer DSH plugin

This independent npm workspace builds one private DSH bundle (`@tyloinfer/dsh-plugin`) for DSH `0.2.2-rc.2`. All sources here are TypeScript, matching the official DSH bundles (`src/*.ts` → `lib/*.js`). `npm run build` compiles [`src/host/`](src/host/) with `tsc` to `lib/host/` and bundles the browser entry from [`src/client/`](src/client/) to `lib/client.js`. DSH loads `lib/host/index.js` and `lib/client.js`; nothing here runs TypeScript at runtime.

From the repository root:

```sh
pnpm --dir app install
pnpm --dir app build:plugins
pnpm --dir app typecheck:plugins
```

The build uses the `app/` workspace dependencies. It reads the shared presentation components and styles from [`../shared/`](../shared/) through the `@shared` alias and the product version from `../server/_version.py`; no standalone Studio dependency installation or build is required. See [Agent Host setup](../../docs/agent-host.md) to run the bundle.

Tests sit next to the source they cover: [`src/host/tests/`](src/host/tests/) and [`src/client/tests/`](src/client/tests/). Host tests import the compiled `lib/host/` output, so build before testing. The `*-smoke.ts` scripts in [`src/host/tests/`](src/host/tests/) need a running core API and are driven by the bridge tests under `../server/tests/`, not by `node --test`.

`pnpm --dir app typecheck:plugins` covers the Host (`tsconfig.host.json`) and the Client logic modules plus their tests (`tsconfig.client.json`: `src/client/*.ts`, `src/client/tests`). The Client `.tsx` page components are still unannotated — their props are implicit `any` — so they are excluded from that program; they compile through Vite, and the `app/shared/` components they render are typechecked by `app/web` itself.
