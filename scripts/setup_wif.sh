#!/usr/bin/env bash
# ===========================================================================
# AgriStress — ONE-TIME keyless CI/CD bootstrap (Workload Identity Federation).
#
# Run this ONCE in Google Cloud Shell (gcloud is preinstalled + authenticated).
# It wires GitHub Actions to deploy to Cloud Run WITHOUT any service-account
# JSON key: GitHub mints a short-lived OIDC token, and WIF exchanges it for
# Google credentials scoped to this one repository.
#
#   bash scripts/setup_wif.sh
#
# What it creates (every step is idempotent — safe to re-run):
#   - enables iamcredentials / run / artifactregistry / cloudbuild APIs
#   - Artifact Registry docker repo  'agristress'         (asia-east1)
#   - deployer service account       'github-deployer'
#       roles: roles/run.admin, roles/artifactregistry.writer,
#              roles/iam.serviceAccountUser   (project-level)
#   - Workload Identity pool         'github-pool'
#   - OIDC provider                  'github-provider'  (GitHub Actions issuer),
#       restricted to this repository via an attribute condition
#   - binds roles/iam.workloadIdentityUser on the SA for this repo's principalSet
#
# At the end it PRINTS the exact GitHub repository Variables to add.
#
# Knobs (env var = default):
#   PROJECT_ID = dmjone                          GCP project.
#   REPO       = divyamohan1993/bah2026-ps6      GitHub owner/repo (OIDC subject).
#   REGION     = asia-east1                       Artifact Registry region.
# ===========================================================================
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-dmjone}"
REPO="${REPO:-divyamohan1993/bah2026-ps6}"
REGION="${REGION:-asia-east1}"

# Fixed names (match .github/workflows/deploy-cloudrun.yml + docs).
GAR_REPO="agristress"
SA_NAME="github-deployer"
POOL_ID="github-pool"
PROVIDER_ID="github-provider"
ISSUER_URI="https://token.actions.githubusercontent.com"

SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

log()  { printf '\033[1;34m[wif]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[wif] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

command -v gcloud >/dev/null 2>&1 || die "gcloud not found. Run this in Google Cloud Shell (https://shell.cloud.google.com)."

log "Project : ${PROJECT_ID}"
log "Repo    : ${REPO}"
log "Region  : ${REGION}"

# ---- 0. select project + resolve its number -------------------------------
gcloud config set project "${PROJECT_ID}" >/dev/null
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
[[ -n "${PROJECT_NUMBER}" ]] || die "Could not resolve the project number for ${PROJECT_ID}."
log "Project number: ${PROJECT_NUMBER}"

# ---- 1. enable required APIs (idempotent) ---------------------------------
log "Enabling APIs (iamcredentials, run, artifactregistry, cloudbuild) ..."
gcloud services enable \
  iamcredentials.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  --project "${PROJECT_ID}"

# ---- 2. Artifact Registry docker repo (describe-or-create) ----------------
if gcloud artifacts repositories describe "${GAR_REPO}" \
      --location="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  log "Artifact Registry repo '${GAR_REPO}' already exists in ${REGION}."
else
  log "Creating Artifact Registry repo '${GAR_REPO}' in ${REGION} ..."
  gcloud artifacts repositories create "${GAR_REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --project="${PROJECT_ID}" \
    --description="AgriStress container images"
fi

# ---- 3. deployer service account (describe-or-create) ---------------------
if gcloud iam service-accounts describe "${SA_EMAIL}" \
      --project="${PROJECT_ID}" >/dev/null 2>&1; then
  log "Service account '${SA_EMAIL}' already exists."
else
  log "Creating service account '${SA_NAME}' ..."
  gcloud iam service-accounts create "${SA_NAME}" \
    --project="${PROJECT_ID}" \
    --display-name="GitHub Actions deployer (WIF, keyless)"
fi

# ---- 4. grant project-level roles to the deployer SA ----------------------
# `add-iam-policy-binding` is idempotent (a duplicate binding is a no-op).
log "Granting roles to ${SA_EMAIL} (run.admin, artifactregistry.writer, iam.serviceAccountUser) ..."
for ROLE in roles/run.admin roles/artifactregistry.writer roles/iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="${ROLE}" \
    --condition=None >/dev/null
  log "  granted ${ROLE}"
done

# ---- 5. Workload Identity pool (describe-or-create) -----------------------
if gcloud iam workload-identity-pools describe "${POOL_ID}" \
      --project="${PROJECT_ID}" --location="global" >/dev/null 2>&1; then
  log "Workload Identity pool '${POOL_ID}' already exists."
else
  log "Creating Workload Identity pool '${POOL_ID}' ..."
  gcloud iam workload-identity-pools create "${POOL_ID}" \
    --project="${PROJECT_ID}" \
    --location="global" \
    --display-name="GitHub Actions pool"
fi

# ---- 6. OIDC provider for GitHub Actions (describe-or-create) --------------
# Restrict to THIS repository so only its workflows can impersonate the SA.
if gcloud iam workload-identity-pools providers describe "${PROVIDER_ID}" \
      --project="${PROJECT_ID}" --location="global" \
      --workload-identity-pool="${POOL_ID}" >/dev/null 2>&1; then
  log "OIDC provider '${PROVIDER_ID}' already exists."
else
  log "Creating OIDC provider '${PROVIDER_ID}' (issuer: GitHub Actions, repo-restricted) ..."
  gcloud iam workload-identity-pools providers create-oidc "${PROVIDER_ID}" \
    --project="${PROJECT_ID}" \
    --location="global" \
    --workload-identity-pool="${POOL_ID}" \
    --display-name="GitHub Actions provider" \
    --issuer-uri="${ISSUER_URI}" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
    --attribute-condition="assertion.repository=='${REPO}'"
fi

# ---- 7. let this repo's principalSet impersonate the deployer SA ----------
# Binding is idempotent. The principalSet matches any workflow run whose OIDC
# token carries attribute.repository == this repo.
PRINCIPAL_SET="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/attribute.repository/${REPO}"
log "Binding roles/iam.workloadIdentityUser on ${SA_EMAIL} for repo ${REPO} ..."
gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
  --project="${PROJECT_ID}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="${PRINCIPAL_SET}" >/dev/null

# ---- 8. print the GitHub repository Variables to add ----------------------
WIF_PROVIDER="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/providers/${PROVIDER_ID}"

printf '\n'
printf '\033[1;32m============================================================================\033[0m\n'
printf '\033[1;32m ADD THESE GITHUB REPOSITORY VARIABLES\033[0m\n'
printf '\033[1;32m (Settings -> Secrets and variables -> Actions -> Variables)\033[0m\n'
printf '\033[1;32m============================================================================\033[0m\n'
printf '  GCP_PROJECT_ID=%s\n'      "${PROJECT_ID}"
printf '  GCP_REGION=%s\n'          "${REGION}"
printf '  GCP_WIF_PROVIDER=%s\n'    "${WIF_PROVIDER}"
printf '  GCP_DEPLOY_SA=%s\n'       "${SA_EMAIL}"
printf '  GAR_REPO=%s\n'            "${GAR_REPO}"
printf '  CLOUD_RUN_SERVICE=%s\n'   "agristress-api"
printf '\033[1;32m============================================================================\033[0m\n'
printf '\nThese are non-secret Variables (not Secrets): no JSON key exists. After\n'
printf 'adding them, trigger the first deploy by pushing to a deploy branch or via\n'
printf 'Actions -> "Deploy to Cloud Run" -> Run workflow (workflow_dispatch).\n'
