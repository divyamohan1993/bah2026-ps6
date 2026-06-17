# CI/CD — Keyless Auto-Deploy to Cloud Run (Workload Identity Federation)

This repo deploys the AgriStress serving image to **Google Cloud Run**
automatically on every push to a deploy branch — **without storing any
service-account JSON key**. Auth is keyless via **GitHub OIDC + Workload
Identity Federation (WIF)**.

- Workflow: [`.github/workflows/deploy-cloudrun.yml`](../.github/workflows/deploy-cloudrun.yml)
- One-time bootstrap: [`scripts/setup_wif.sh`](../scripts/setup_wif.sh)
- Manual / first deploy fallback: [`scripts/deploy_cloudrun.sh`](../scripts/deploy_cloudrun.sh)

Fixed parameters: project `dmjone`, region `asia-east1`, service
`agristress-api`, Artifact Registry repo `agristress`, scale-to-zero
(`--min-instances=0 --max-instances=1`).

---

## 1. What is WIF, and why keyless?

A long-lived service-account **JSON key** is a credential you must store as a
GitHub Secret, rotate, and protect — if it leaks, anyone can act as that SA.

**Workload Identity Federation** removes the key entirely:

1. On each run, GitHub Actions mints a short-lived **OIDC token** that asserts
   *which repository* (and ref/workflow) is running.
2. Google's **Workload Identity Provider** verifies that token against the
   GitHub Actions issuer (`https://token.actions.githubusercontent.com`) and an
   **attribute condition** that restricts it to **this repository only**.
3. WIF then lets that identity **impersonate** the deployer service account
   (`github-deployer@dmjone.iam.gserviceaccount.com`) for the duration of the
   job — minted on demand, auto-expiring, nothing stored.

Net: no key to leak or rotate, and only `divyamohan1993/bah2026-ps6` can obtain
the deploy credentials.

---

## 2. One-time bootstrap (run once in Cloud Shell)

Run this **once**, in [Google Cloud Shell](https://shell.cloud.google.com)
(gcloud is preinstalled and already authenticated as you):

```bash
git clone -b claude/keen-lovelace-nmrhjw https://github.com/divyamohan1993/bah2026-ps6.git
cd bah2026-ps6
bash scripts/setup_wif.sh
```

It is **idempotent** (safe to re-run). It will:

- enable the `iamcredentials`, `run`, `artifactregistry`, and `cloudbuild` APIs;
- create the Artifact Registry docker repo `agristress` in `asia-east1`;
- create the deployer service account `github-deployer` and grant it
  `roles/run.admin`, `roles/artifactregistry.writer`, and
  `roles/iam.serviceAccountUser` at the project level;
- create the Workload Identity **pool** `github-pool` and the OIDC **provider**
  `github-provider`, restricted to this repository via an attribute condition;
- bind `roles/iam.workloadIdentityUser` on the SA for this repo's principalSet.

At the end it prints the exact repository **Variables** to add (with your real
project number substituted).

> Override defaults if needed: `PROJECT_ID=... REPO=owner/repo REGION=...
> bash scripts/setup_wif.sh`.

---

## 3. Add the GitHub repository Variables

In GitHub: **Settings → Secrets and variables → Actions → Variables** (the
**Variables** tab, not Secrets — none of these is secret), add the six values
that `setup_wif.sh` printed:

| Variable | Value |
|---|---|
| `GCP_PROJECT_ID` | `dmjone` |
| `GCP_REGION` | `asia-east1` |
| `GCP_WIF_PROVIDER` | `projects/<PROJECT_NUMBER>/locations/global/workloadIdentityPools/github-pool/providers/github-provider` |
| `GCP_DEPLOY_SA` | `github-deployer@dmjone.iam.gserviceaccount.com` |
| `GAR_REPO` | `agristress` |
| `CLOUD_RUN_SERVICE` | `agristress-api` |

`<PROJECT_NUMBER>` is your project's numeric id — `setup_wif.sh` substitutes it
for you in the printed block (e.g. `projects/123456789012/...`).

---

## 4. How the workflow triggers

[`.github/workflows/deploy-cloudrun.yml`](../.github/workflows/deploy-cloudrun.yml)
runs on:

- **push** to `main` or `claude/keen-lovelace-nmrhjw` (the feature branch — a
  `TODO` comment marks it for removal after merge), and
- **`workflow_dispatch`** (manual run from the Actions tab).

It has `permissions: { contents: read, id-token: write }` — `id-token: write`
is **required** for WIF. The job then:

1. `actions/checkout@v4`
2. `google-github-actions/auth@v2` (keyless, using the three `vars.GCP_*`
   values)
3. `google-github-actions/setup-gcloud@v2`
4. `gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet`
5. `docker build -f infra/Dockerfile -t $IMAGE .` where
   `IMAGE = ${REGION}-docker.pkg.dev/${PROJECT}/${GAR_REPO}/${SERVICE}:${{ github.sha }}`
6. `docker push $IMAGE`
7. `gcloud run deploy ${SERVICE} --image $IMAGE --region ${REGION}
   --min-instances=0 --max-instances=1 --cpu-boost --memory=512Mi --cpu=1
   --port=8080 --allow-unauthenticated`
8. prints the service URL to the job summary.

Every config value comes from non-secret repository **Variables** (`vars.*`).

---

## 5. The very first deploy

Once the Variables are set, deploy in any one of these ways:

- **Push** to `main` or `claude/keen-lovelace-nmrhjw` — the workflow runs
  automatically.
- **Manual**: GitHub → **Actions → "Deploy to Cloud Run" → Run workflow**
  (`workflow_dispatch`).
- **From Cloud Shell** (bypasses CI, uses Cloud Build):
  `bash scripts/deploy_cloudrun.sh` — same scale-to-zero profile, prints the URL
  and runs a `/health` smoke check.

After it deploys, the service URL is in the workflow's job summary; check
`${URL}/health`.

---

## 6. Cleanup after merge

When `claude/keen-lovelace-nmrhjw` is merged into `main`, delete the feature
branch from the `on.push.branches` list in the workflow (a `TODO` comment marks
the exact line) so only `main` auto-deploys.

See also [`docs/DEPLOYMENT.md`](./DEPLOYMENT.md) (full Cloud Run guide) and
[`docs/CUSTOM_DOMAIN.md`](./CUSTOM_DOMAIN.md) (custom-domain mapping).
