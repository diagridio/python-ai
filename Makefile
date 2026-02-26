# Test targets
.PHONY: test
test:
	@echo "Running tests..."
	uv run pytest tests/ -v --tb=short

.PHONY: test-cov
test-cov:
	@echo "Running tests with coverage..."
	uv run pytest tests/ -v --cov=diagrid --cov-report=term-missing --cov-report=html

.PHONY: test-install
test-install:
	@echo "Installing test dependencies..."
	uv sync --group test

.PHONY: test-all
test-all: test-install test-cov
	@echo "All tests completed!"

# Pre-commit hook targets
.PHONY: hooks-install
hooks-install:
	@echo "Installing pre-push hooks..."
	pre-commit install --hook-type pre-push

.PHONY: hooks-uninstall
hooks-uninstall:
	@echo "Uninstalling pre-push hooks..."
	pre-commit uninstall --hook-type pre-push

.PHONY: hooks-run
hooks-run:
	@echo "Running all pre-push hooks..."
	pre-commit run --all-files --hook-stage pre-push

# .PHONY: hooks-run-all
# hooks-run-all:
# 	@echo "Running all pre-push hooks plus integration tests..."
# 	@echo "Step 1/2: Running pre-push hooks (format, lint, type check, unit tests)..."
# 	pre-commit run --all-files --hook-stage pre-push
# 	@echo "Step 2/2: Running integration tests..."
# 	uv run pytest tests -m integration -v

.PHONY: format
format:
	@echo "Formatting code with ruff..."
	uv run ruff format diagrid tests

.PHONY: lint
lint:
	@echo "Linting with flake8..."
	uv run flake8 diagrid tests --ignore=E501,F401,W503,E203,E704

.PHONY: typecheck
typecheck:
	@echo "Type checking with mypy..."
	uv run mypy --config-file mypy.ini

.PHONY: test-unit
test-unit:
	@echo "Running unit tests..."
	uv run pytest tests -m "not integration" -v

# --- Configuration ---
CLUSTER_NAME ?= catalyst-agents
REG_PORT     ?= 5001
REG_NAME     ?= kind-registry
REG_IMAGE    ?= registry:2
KIND_CONFIG  ?= charts/kind-config.yaml

TEST_CLUSTER_NAME ?= catalyst-agents-test
TEST_KIND_CONFIG  ?= charts-tests/kind-config-test.yaml
MIRROR_CACHE      ?= /tmp/registry-mirror-cache

# List of registries to mirror: <host>,<remote_url>
# Using comma as separator to avoid shell pipe issues.
MIRRORS = docker.io,https://registry-1.docker.io \
          ghcr.io,https://ghcr.io \
          quay.io,https://quay.io \
          gcr.io,https://gcr.io \
          registry.k8s.io,https://registry.k8s.io

# ---------------------------------------------------------------------------
# Mirror helpers (shared by cluster-up and test-chainsaw)
# ---------------------------------------------------------------------------
define start-mirrors
	@for mirror in $(MIRRORS); do \
		host=`echo $$mirror | cut -d',' -f1`; \
		url=`echo $$mirror | cut -d',' -f2`; \
		name=`echo $$host | sed 's/\./-/g'`; \
		reg_mirror_name="kind-mirror-$$name"; \
		cache_dir="$(1)/$$name"; \
		mkdir -p "$$cache_dir"; \
		if [ "`docker inspect -f '{{.State.Running}}' $$reg_mirror_name 2>/dev/null || true`" != "true" ]; then \
			echo "-> Starting mirror for $$host"; \
			docker rm -f "$$reg_mirror_name" 2>/dev/null || true; \
			docker run -d --restart=always --network bridge \
				--name "$$reg_mirror_name" \
				-v "$$cache_dir:/var/lib/registry" \
				-e REGISTRY_PROXY_REMOTEURL=$$url $(REG_IMAGE); \
		fi; \
	done
endef

define configure-mirrors
	@for node in `kind get nodes --name $(1)`; do \
		for mirror in $(MIRRORS); do \
			host=`echo $$mirror | cut -d',' -f1`; \
			name=`echo $$host | sed 's/\./-/g'`; \
			reg_mirror_name="kind-mirror-$$name"; \
			dir="/etc/containerd/certs.d/$$host"; \
			docker exec "$$node" mkdir -p "$$dir"; \
			echo '[host."http://'$$reg_mirror_name':5000"]' \
				| docker exec -i "$$node" cp /dev/stdin "$$dir/hosts.toml"; \
		done; \
	done
endef

define connect-mirrors
	@for mirror in $(MIRRORS); do \
		host=`echo $$mirror | cut -d',' -f1`; \
		name=`echo $$host | sed 's/\./-/g'`; \
		reg_mirror_name="kind-mirror-$$name"; \
		if [ "`docker inspect -f='{{json .NetworkSettings.Networks.kind}}' $$reg_mirror_name 2>/dev/null`" = "null" ]; then \
			docker network connect "kind" "$$reg_mirror_name"; \
		fi; \
	done
endef

# ---------------------------------------------------------------------------
# Cluster lifecycle
# ---------------------------------------------------------------------------
.PHONY: cluster-up
cluster-up:
	@echo "### Starting local dev registry and mirrors... ###"
	@if [ "`docker inspect -f '{{.State.Running}}' $(REG_NAME) 2>/dev/null || true`" != "true" ]; then \
		docker run -d --restart=always -p "127.0.0.1:$(REG_PORT):5000" --network bridge --name "$(REG_NAME)" $(REG_IMAGE); \
	else \
		docker start $(REG_NAME) 2>/dev/null || true; \
	fi
	$(call start-mirrors,/tmp/dev-mirror-cache)
	@echo "### Creating kind cluster... ###"
	@kind create cluster --name $(CLUSTER_NAME) --config $(KIND_CONFIG)
	@echo "### Configuring registry access on nodes... ###"
	@for node in `kind get nodes --name $(CLUSTER_NAME)`; do \
		dir="/etc/containerd/certs.d/localhost:$(REG_PORT)"; \
		docker exec "$$node" mkdir -p "$$dir"; \
		echo '[host."http://$(REG_NAME):5000"]' | docker exec -i "$$node" cp /dev/stdin "$$dir/hosts.toml"; \
	done
	$(call configure-mirrors,$(CLUSTER_NAME))
	@echo "### Connecting registries to cluster network... ###"
	@if [ "`docker inspect -f='{{json .NetworkSettings.Networks.kind}}' $(REG_NAME)`" = "null" ]; then \
		docker network connect "kind" "$(REG_NAME)"; \
	fi
	$(call connect-mirrors)
	@echo "### Documenting local registry... ###"
	@printf "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: local-registry-hosting\n  namespace: kube-public\ndata:\n  localRegistryHosting.v1: |\n    host: \"localhost:$(REG_PORT)\"\n    help: \"https://kind.sigs.k8s.io/docs/user/local-registry/\"\n" | kubectl apply -f -

.PHONY: cluster-down
cluster-down:
	@echo "### Tearing down environment... ###"
	@kind delete cluster --name $(CLUSTER_NAME)

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

.PHONY: helm-install
helm-install:
	@echo "### Installing stack... ###"
	@helm dependency update charts/catalyst-agents
	@helm upgrade --install catalyst-agents ./charts/catalyst-agents \
		--namespace catalyst-agents --create-namespace \
		--set llm.apiKey=$(OPENAI_API_KEY)

.PHONY: helm-test-lint
helm-test-lint:
	@echo "### Running helm lint... ###"
	@helm lint ./charts/catalyst-agents --set llm.apiKey=lint-key --set monitoring.enabled=false --set kagent.enabled=false
	@helm lint ./charts/catalyst-agents --set llm.apiKey=lint-key

.PHONY: helm-test-template
helm-test-template:
	@echo "### Rendering and validating templates... ###"
	@helm template catalyst-agents ./charts/catalyst-agents \
		--namespace catalyst-agents \
		--set monitoring.enabled=false \
		--set kagent.enabled=false \
		--set dapr.enabled=false \
		--set llm.apiKey=test-key \
		| kubeconform -strict -summary -schema-location default -skip CustomResourceDefinition,Component,Configuration,HTTPEndpoint,Agent,Memory,ModelConfig,RemoteMCPServer,ToolServer,MCPServer

.PHONY: helm-test-chainsaw
helm-test-chainsaw:
	@echo "### Running Chainsaw integration tests... ###"
	@helm dependency update ./charts/catalyst-agents
	@echo "### Starting registry mirrors... ###"
	$(call start-mirrors,$(MIRROR_CACHE))
	@echo "### Creating Kind cluster... ###"
	@kind create cluster --name $(TEST_CLUSTER_NAME) --config $(TEST_KIND_CONFIG) || true
	$(call configure-mirrors,$(TEST_CLUSTER_NAME))
	$(call connect-mirrors)
	@echo "### Installing chart... ###"
	@helm upgrade --install catalyst-agents ./charts/catalyst-agents \
		--namespace catalyst-agents --create-namespace \
		--set llm.apiKey=ci-test-key \
		--kube-context kind-$(TEST_CLUSTER_NAME) \
		--timeout 10m
	@chainsaw test charts-tests/chainsaw/ \
		--parallel 1 \
		--report-format JSON \
		--report-name chainsaw-results \
		--kube-context kind-$(TEST_CLUSTER_NAME); \
		exit_code=$$?; \
		kind delete cluster --name $(TEST_CLUSTER_NAME); \
		exit $$exit_code

.PHONY: helm-test
helm-test: helm-test-lint helm-test-template helm-test-chainsaw

.PHONY: all
all: cluster-up helm-install
