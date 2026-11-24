# Shortcuts for native builds, the app workspace and desktop packaging.
# Options and targets are documented in docs/build.md and docs/packaging.md.

BUILD_DIR  ?= build
BUILD_TYPE ?= Release
JOBS       ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu)
SERVER_VENV := app/server/.venv

ifeq ($(shell uname -s),Darwin)
  # CMake looks for MLX under <repo>/.venv; uv keeps it in the server project.
  TARGET    := mfq-decode-metal
  RUNTIME   := $(BUILD_DIR)/metal/$(TARGET)
  UV_EXTRA  := --extra metal
  CONFIGURE = -DTYLOINFER_MLX_ROOT="$(shell uv run --project app/server python -c 'import mlx,os;print(os.path.dirname(mlx.__file__))')"
else
  TARGET    := mfq-runtime
  RUNTIME   := $(BUILD_DIR)/$(TARGET)
  UV_EXTRA  :=
  CONFIGURE :=
endif

.PHONY: all help configure engine server ui dev build package serve test clean

PARTS := engine server ui

all: help

help:
	@echo "make build                       all three parts below"
	@echo "make build engine|server|ui      one part: C++ runtime / Python env / Web UI"
	@echo "make serve                       start Studio on http://127.0.0.1:8090 (needs make build)"
	@echo "make engine R=<target>           build one native target, e.g. a test binary"
	@echo "make test engine                 build and run the native C++ tests (R=<target> for one)"
	@echo "make dev                         all three parts, Python with the dev extra"
	@echo "make package studio|agent        package one app (default: Studio)"
	@echo "make package agent -- --clean    clean rebuild of Agent"
	@echo "make clean                       remove $(BUILD_DIR)/"

$(BUILD_DIR)/CMakeCache.txt:
	cmake -S csrc -B $(BUILD_DIR) -G Ninja -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(CONFIGURE)

configure: $(BUILD_DIR)/CMakeCache.txt
# The Apple configure line probes this environment for MLX, so it must exist first.
configure: $(SERVER_VENV)/pyvenv.cfg

# The native inference runtime; R=<target> builds that one target instead.
engine: configure
	cmake --build $(BUILD_DIR) --target $(or $(R),$(TARGET)) --parallel $(JOBS)

# Everything a local run needs; serve depends on none of it. A named part makes
# make build <part> build just that part.
build: $(or $(filter $(PARTS),$(MAKECMDGOALS)),$(PARTS))

# uv run would sync anyway; the target keeps the dependency explicit and re-runs
# only when the lockfile or pyproject.toml changed. UV_EXTRA carries what the
# host needs (Apple: MLX for the Metal runtime).
$(SERVER_VENV)/pyvenv.cfg: app/server/pyproject.toml app/server/uv.lock
	uv sync --project app/server $(UV_EXTRA)

server: $(SERVER_VENV)/pyvenv.cfg

# The development environment: the other two parts, and one Python sync with both
# the host extra and the dev extra. uv creates the venv when it is missing, so the
# plain sync above is not a prerequisite; running it first would only prune dev.
dev: engine ui
	uv sync --project app/server --extra dev $(UV_EXTRA)

# The same two commands python -m serve runs when app/web/dist is out of date.
ui:
	pnpm --dir app install --frozen-lockfile
	pnpm --dir app run build:studio

package:
	cd app && pnpm run package --app "$(or $(filter studio agent,$(MAKECMDGOALS)),studio)" $(if $(filter --clean,$(MAKECMDGOALS)),--clean)

# Start only; run make build first so the runtime, Python env and Web UI exist.
serve:
	uv run --project app/server python -m serve --running-executable $(RUNTIME)

# make test engine runs the native tests: the full build covers every test
# executable CMake configured. R=<target> builds and runs that one only; engine
# then rebuilds the same target, which ninja reports as no work.
test:
	cmake --build $(BUILD_DIR) $(if $(R),--target $(R),) --parallel $(JOBS)
	ctest --test-dir $(BUILD_DIR) -R '$(or $(R),.)' --output-on-failure

# With package, these goals are arguments, not separate build or cleanup jobs.
ifneq ($(filter package,$(MAKECMDGOALS)),)
.PHONY: studio agent --clean
studio agent --clean:
	@:
ifneq ($(filter clean,$(MAKECMDGOALS)),)
$(error Use 'make package $(or $(filter studio agent,$(MAKECMDGOALS)),studio) -- --clean')
endif
else
clean:
	rm -rf $(BUILD_DIR)
endif
