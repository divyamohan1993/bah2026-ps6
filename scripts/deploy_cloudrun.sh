#!/usr/bin/env bash
# ===========================================================================
# AgriStress — one-command Cloud Run deploy bootstrap.
#
# Builds the serving image in Cloud Build (NO local Docker required) from
# infra/Dockerfile, pushes it to Artifact Registry, and deploys it to Cloud Run
# with the scale-to-zero profile (min=0/max=1 by default). Then prints the URL
# and runs a /health smoke check.
#
# Usage (env or args; all have defaults — just run it):
#   bash scripts/deploy_cloudrun.sh
#   PROJECT_ID=other-project REGION=asia-east1 bash scripts/deploy_cloudrun.sh
#
# Cloud Shell one-liner (nothing to install — gcloud + curl are preinstalled):
#   git clone -b claude/keen-lovelace-nmrhjw https://github.com/divyamohan1993/bah2026-ps6.git \
#     && cd bah2026-ps6 \
#     && bash scripts/deploy_cloudrun.sh
#
# Knobs (env var = default):
#   PROJECT_ID    = dmjone               GCP project to deploy into.
#   REGION        = asia-east1           Cloud Run + Artifact Registry region.
#   SERVICE       = agristress-api       Cloud Run service name.
#   REPO          = agristress           Artifact Registry docker repo name.
#   MIN_INSTANCES = 0                    Scale-to-zero when idle (set 1 = warm).
#   MAX_INSTANCES = 1                    Instance cap.
#
# Idempotent: safe to re-run. It enables APIs and creates the Artifact Registry
# repo only if missing, then submits the build each time to ship a new revision.
# ===========================================================================
set -euo pipefail

# ---- inputs (env wins; sane defaults) -------------------------------------
PROJECT_ID="${PROJECT_ID:-dmjone}"
REGION="${REGION:-asia-east1}"
SERVICE="${SERVICE:-agristress-api}"
REPO="${REPO:-agristress}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
MAX_INSTANCES="${MAX_INSTANCES:-1}"

# Resolve repo root from this script's location so it works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CLOUDBUILD_CONFIG="${REPO_ROOT}/cloudbuild.yaml"

log()  { printf '\033[1;34m[deploy]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[deploy] WARN:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[deploy] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ---- 0. validate inputs / environment -------------------------------------
if [[ -z "${PROJECT_ID}" ]]; then
  die "PROJECT_ID resolved empty. It defaults to 'dmjone'; only override it with
      a non-empty value, e.g.: PROJECT_ID=my-project bash scripts/deploy_cloudrun.sh
      The project must have billing enabled."
fi

if ! command -v gcloud >/dev/null 2>&1; then
  die "gcloud CLI not found. The easiest fix is to run this in Google Cloud
      Shell (https://shell.cloud.google.com) where gcloud + curl are
      preinstalled and you are already authenticated. Otherwise install the
      Google Cloud SDK and run 'gcloud auth login' first."
fi

if [[ ! -f "${CLOUDBUILD_CONFIG}" ]]; then
  die "cloudbuild.yaml not found at ${CLOUDBUILD_CONFIG}. Run this script from a
      checkout of the repo (it builds from the repo root)."
fi

log "Project        : ${PROJECT_ID}"
log "Region         : ${REGION}"
log "Service        : ${SERVICE}"
log "Artifact repo  : ${REPO}"
log "Scaling        : min=${MIN_INSTANCES} max=${MAX_INSTANCES} (min=0 => scale-to-zero)"
log "Build context  : ${REPO_ROOT}"

# ---- 1. select the project ------------------------------------------------
log "Setting active project to ${PROJECT_ID} ..."
gcloud config set project "${PROJECT_ID}" >/dev/null

# ---- 2. enable required APIs (idempotent) ---------------------------------
log "Enabling required services (run, cloudbuild, artifactregistry) ..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  --project "${PROJECT_ID}"

# ---- 3. ensure the Artifact Registry docker repo exists -------------------
if gcloud artifacts repositories describe "${REPO}" \
      --location="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  log "Artifact Registry repo '${REPO}' already exists in ${REGION}."
else
  log "Creating Artifact Registry repo '${REPO}' in ${REGION} ..."
  gcloud artifacts repositories create "${REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --project="${PROJECT_ID}" \
    --description="AgriStress container images"
fi

# ---- 4. build + push + deploy via Cloud Build -----------------------------
# cloudbuild.yaml builds with `-f infra/Dockerfile` (context = repo root), so the
# correct image is built every time — no buildpacks fallback, no local Docker.
log "Submitting Cloud Build (build infra/Dockerfile -> push -> deploy) ..."
gcloud builds submit "${REPO_ROOT}" \
  --config "${CLOUDBUILD_CONFIG}" \
  --project "${PROJECT_ID}" \
  --substitutions="_PROJECT=${PROJECT_ID},_REGION=${REGION},_SERVICE=${SERVICE},_REPO=${REPO},_MIN_INSTANCES=${MIN_INSTANCES},_MAX_INSTANCES=${MAX_INSTANCES}"

# ---- 5. fetch the service URL + smoke-check /health -----------------------
log "Fetching the Cloud Run service URL ..."
URL="$(gcloud run services describe "${SERVICE}" \
  --region="${REGION}" --project="${PROJECT_ID}" \
  --format='value(status.url)')"

if [[ -z "${URL}" ]]; then
  warn "Could not read the service URL (the deploy may still have succeeded)."
else
  if command -v curl >/dev/null 2>&1; then
    log "Smoke-checking ${URL}/health ..."
    if curl -fsS "${URL}/health"; then
      printf '\n'
      log "Health check OK."
    else
      warn "Health check did not return success yet — the first request after a"
      warn "cold start can take a few seconds. Retry: curl -fsS ${URL}/health"
    fi
  else
    warn "curl not found; skipping smoke check. Try: ${URL}/health"
  fi
fi

# ---- 6. final banner ------------------------------------------------------
printf '\n'
printf '\033[1;32m============================================================\033[0m\n'
printf '\033[1;32m Deployed: %s\033[0m\n' "${URL:-<unknown — check the Cloud Run console>}"
printf '\033[1;32m============================================================\033[0m\n'
if [[ "${MIN_INSTANCES}" == "0" ]]; then
  log "Scale-to-zero is ON (min=0): the service runs only when called and bills"
  log "~\$0 while idle. The first request after idle pays a brief cold start."
  log "Want an always-warm instance? Re-run with MIN_INSTANCES=1."
else
  log "Always-warm (min=${MIN_INSTANCES}): no cold start, small continuous cost."
fi
log "Useful endpoints: ${URL:-<URL>}/health  ${URL:-<URL>}/aoi"
