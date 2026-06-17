#!/usr/bin/env bash
# ===========================================================================
# AgriStress — map a custom domain to the Cloud Run service (zero-cost).
#
# Creates a Cloud Run **domain mapping** for a single subdomain (default
# agristress.dmj.one). A domain mapping is FREE and keeps the service
# scale-to-zero + public — no Load Balancer, no fixed cost. It then prints the
# DNS record(s) you must add at your registrar, and Google provisions a managed
# SSL cert automatically once DNS resolves.
#
#   bash scripts/map_domain.sh
#   DOMAIN=agristress.example.org bash scripts/map_domain.sh
#
# PREREQUISITE (one-time, manual): you must have **verified ownership** of the
# domain first, otherwise the mapping is rejected. Verify via:
#   gcloud domains verify dmj.one
# (opens Google Search Console / Webmaster Central) — add the TXT record it
# gives you at the dmj.one registrar, confirm, then re-run this script. This is
# a manual, registrar-side step that cannot be automated from here.
#
# Idempotent: if the mapping already exists it is left as-is and its DNS records
# are printed.
#
# Knobs (env var = default):
#   DOMAIN  = agristress.dmj.one     The custom (sub)domain to map.
#   SERVICE = agristress-api          Cloud Run service to map it to.
#   REGION  = asia-east1              Cloud Run region (must support mappings).
#   PROJECT_ID = dmjone               GCP project.
# ===========================================================================
set -euo pipefail

DOMAIN="${DOMAIN:-agristress.dmj.one}"
SERVICE="${SERVICE:-agristress-api}"
REGION="${REGION:-asia-east1}"
PROJECT_ID="${PROJECT_ID:-dmjone}"

log()  { printf '\033[1;34m[domain]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[domain] WARN:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[domain] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

command -v gcloud >/dev/null 2>&1 || die "gcloud not found. Run this in Google Cloud Shell (https://shell.cloud.google.com)."

log "Domain  : ${DOMAIN}"
log "Service : ${SERVICE}"
log "Region  : ${REGION}"
log "Project : ${PROJECT_ID}"

gcloud config set project "${PROJECT_ID}" >/dev/null

# ---- 1. create the domain mapping (describe-or-create) --------------------
if gcloud beta run domain-mappings describe \
      --domain "${DOMAIN}" \
      --region "${REGION}" \
      --project "${PROJECT_ID}" >/dev/null 2>&1; then
  log "Domain mapping for ${DOMAIN} already exists."
else
  log "Creating domain mapping ${DOMAIN} -> ${SERVICE} ..."
  if ! gcloud beta run domain-mappings create \
        --service "${SERVICE}" \
        --domain "${DOMAIN}" \
        --region "${REGION}" \
        --project "${PROJECT_ID}"; then
    warn "Create failed. The most common cause is unverified domain ownership."
    warn "Verify it first (one-time):  gcloud domains verify ${DOMAIN#*.}"
    warn "then re-run:  bash scripts/map_domain.sh"
    die  "Domain mapping not created."
  fi
fi

# ---- 2. print the DNS records the user must add ---------------------------
log "DNS records to add at your registrar for ${DOMAIN}:"
echo "----------------------------------------------------------------------"
gcloud beta run domain-mappings describe \
  --domain "${DOMAIN}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --format='value(status.resourceRecords)'
echo "----------------------------------------------------------------------"

cat <<EOF

Next steps (manual, registrar-side):
  1. Add the record(s) above at the dmj.one DNS provider. For a subdomain like
     '${DOMAIN}' this is typically a CNAME:
         ${DOMAIN%%.*}  CNAME  ghs.googlehosted.com.
     (An apex/root domain instead needs the A/AAAA records shown above.)
  2. Wait for DNS to propagate. Google then auto-issues a managed SSL
     certificate — this can take from a few minutes up to ~24h.
  3. Verify:  curl -fsS https://${DOMAIN}/health

The Cloud Run service stays --allow-unauthenticated and scale-to-zero behind the
domain — the mapping adds no fixed cost.
EOF
