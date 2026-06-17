# AgriStress — developer & deployment shortcuts.
# See docs/DEPLOYMENT.md for the full Google Cloud Run guide.

# ---- configurable knobs (override on the command line) --------------------
PROJECT_ID  ?= $(shell gcloud config get-value project 2>/dev/null)
REGION      ?= asia-south1
SERVICE     ?= agristress-api
PORT        ?= 8080
IMAGE       ?= $(REGION)-docker.pkg.dev/$(PROJECT_ID)/agristress/$(SERVICE):$(shell git rev-parse --short HEAD 2>/dev/null || echo latest)

.DEFAULT_GOAL := help
.PHONY: help install test lint serve docker-build docker-run compose-up \
        cloudbuild deploy-cloudrun

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
	gcloud builds submit --config cloudbuild.yaml \
	  --substitutions=_REGION=$(REGION),_SERVICE=$(SERVICE) .

# Deploy straight to Cloud Run from source (Cloud Build builds infra/Dockerfile).
deploy-cloudrun: ## Deploy the serving API to Cloud Run (PROJECT_ID/REGION overridable).
	gcloud run deploy $(SERVICE) \
	  --source . \
	  --project $(PROJECT_ID) \
	  --region $(REGION) \
	  --port $(PORT) \
	  --allow-unauthenticated \
	  --min-instances=1 \
	  --max-instances=10 \
	  --memory=512Mi \
	  --cpu=1
