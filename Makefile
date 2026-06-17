# AgriStress — developer & deployment shortcuts.
# See docs/DEPLOYMENT.md for the full Google Cloud Run guide.

# ---- configurable knobs (override on the command line) --------------------
PROJECT_ID  ?= dmjone
REGION      ?= asia-east1
SERVICE     ?= agristress-api
REPO        ?= agristress
PORT        ?= 8080
IMAGE       ?= $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(REPO)/$(SERVICE):$(shell git rev-parse --short HEAD 2>/dev/null || echo latest)

.DEFAULT_GOAL := help
.PHONY: help install test lint serve docker-build docker-run compose-up \
        cloudbuild deploy-cloudrun deploy-cloudrun-warm

help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install the package with dev/test extras into the active venv.
	pip install -e ".[dev]"

test: ## Run the test suite.
	pytest -q

lint: ## Lint with ruff.
	ruff check src tests

serve: ## Run the serving API locally (honours $PORT, default 8080).
	PORT=$(PORT) agristress serve

docker-build: ## Build the Cloud Run serving image.
	docker build -f infra/Dockerfile -t $(SERVICE):local .

docker-run: docker-build ## Build then run the image on $PORT.
	docker run --rm -e PORT=$(PORT) -p $(PORT):$(PORT) $(SERVICE):local

compose-up: ## Bring up the local API + Redis stack.
	docker compose -f infra/docker-compose.yml up --build

cloudbuild: ## Build + push + deploy via Cloud Build (uses cloudbuild.yaml).
	gcloud builds submit --config cloudbuild.yaml --project $(PROJECT_ID) \
	  --substitutions=_PROJECT=$(PROJECT_ID),_REGION=$(REGION),_SERVICE=$(SERVICE),_REPO=$(REPO) .

# Build + push + deploy via Cloud Build. cloudbuild.yaml builds the image with an
# explicit `-f infra/Dockerfile` (the canonical Dockerfile is NOT at the repo
# root), so there is NO reliance on `gcloud run deploy --source .` discovering a
# root Dockerfile — and therefore no silent buildpacks fallback.
# Default = scale-to-zero, pay-per-use: 0 instances when idle, capped at 1.
deploy-cloudrun: ## Deploy to Cloud Run, scale-to-zero (min=0,max=1, pay-per-use).
	gcloud builds submit --config cloudbuild.yaml --project $(PROJECT_ID) \
	  --substitutions=_PROJECT=$(PROJECT_ID),_REGION=$(REGION),_SERVICE=$(SERVICE),_REPO=$(REPO),_MIN_INSTANCES=0,_MAX_INSTANCES=1 \
	  .

# Always-warm variant (min=1) — no cold start, for judging/demo days.
deploy-cloudrun-warm: ## Deploy to Cloud Run, always-warm (min=1,max=1) for demos.
	gcloud builds submit --config cloudbuild.yaml --project $(PROJECT_ID) \
	  --substitutions=_PROJECT=$(PROJECT_ID),_REGION=$(REGION),_SERVICE=$(SERVICE),_REPO=$(REPO),_MIN_INSTANCES=1,_MAX_INSTANCES=1 \
	  .
