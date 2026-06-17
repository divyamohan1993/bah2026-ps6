# Deployment Guide — AgriStress on Google Cloud

**Authoritative guide for deploying AgriStress (ISRO BAH 2026 · PS6) to Google
Cloud.** It is accurate to *this* codebase: the FastAPI app
`agristress.serving.api:app`, the `agristress` Typer/argparse CLI, the
`REDIS_URL`-aware cache, and the `[serving]` / `[geo]` / `[cloud]` extras.

> **TL;DR — you do _not_ need a VM.** The whole system runs as managed,
> scale-to-zero services. The serving API + dashboard go on **Cloud Run**, the
> heavy fusion/classification runs **server-side in Google Earth Engine**, batch
> jobs run as **Cloud Run Jobs**, and state lives in **Memorystore + GCS**. A VM
> (or Vertex AI GPU) is only warranted for *optional* persistent deep-learning
> fusion / foundation-model fine-tuning — see [When you need a VM](#8-when-you-do-need-a-vm--vertex-gpu).

---

## 1. Decision table — what runs where

| Component | Where it runs | Why |
|---|---|---|
| **Serving API** (`agristress.serving.api:app`) + **static dashboard** (`dashboard/`) | **Cloud Run** (service) | Stateless HTTP, scale-to-zero, listens on `$PORT`. The O(1) read-hot-path. |
| **Heavy multi-sensor fusion & crop/stress classification** | **Google Earth Engine** (server-side), orchestrated from Cloud Run | EE holds the petabyte optical+SAR archives; compute runs next to the data (`gee/01..06_*.py`). Cloud Run just submits `ee.*` graphs / exports. |
| **Batch / training pipeline** (`agristress ingest \| fuse \| features \| train \| advisory`) | **Cloud Run Jobs** (default) → **Cloud Batch** or **Vertex AI Custom Job** for big/GPU runs | Run-to-completion workloads, not request/response. Same image, different entrypoint. |
| **Dynamic map tiles** | **TiTiler on Cloud Run** (COG reads from GCS) *or* **PMTiles on GCS + Cloud CDN** | TiTiler renders COG tiles on demand (`/cog`, needs `[geo]`); PMTiles is a static, CDN-cached pyramid (cheapest). The repo ships a demo PNG tiler that needs neither. |
| **State / cache** (the serving cache) | **Memorystore for Redis** via `REDIS_URL` | Cloud Run instances are ephemeral; the cache must be external. Falls back to in-memory LRU when `REDIS_URL` is unset. |
| **Gold products** (advisory rasters, rollups, GeoJSON, model artifacts) | **GCS** (objects) + **BigQuery** (tabular rollups / time series) | Durable, queryable, served to the API and the dashboard. |
| **Secrets** (EE service-account JSON, Redis auth) | **Secret Manager** | Mounted into Cloud Run as env/secret, never baked into the image. |
| **Persistent GPU DL fusion / foundation-model fine-tune** (optional, `[dl]`) | **Vertex AI** training job, or a GPU **VM** only if you need a long-lived box | The *only* case that benefits from a VM/GPU. Not required for PS6 deliverables. |

**Net:** every required component is serverless. No VM in the critical path.

---

## 2. Cloud Run constraints & how we satisfy them

| Constraint | How AgriStress satisfies it |
|---|---|
| **Listen on `0.0.0.0:$PORT`** (Cloud Run injects `$PORT`, default `8080`) | `infra/Dockerfile` `CMD` runs `uvicorn agristress.serving.api:app --host 0.0.0.0 --port ${PORT}`. The `agristress serve` CLI also reads `$HOST`/`$PORT` (defaults `0.0.0.0:8080`). |
| **Stateless** (instances are ephemeral, no sticky local disk) | Cache is externalized to Memorystore via `REDIS_URL`; gold products to GCS/BigQuery. The seeded in-memory demo store is for the credentials-free demo only. |
| **Read-only filesystem except `/tmp`** (`/tmp` is in-memory and counts against memory) | The app writes nothing to disk on the request path. Any scratch (e.g. a rendered tile) stays in memory / Redis. |
| **Request timeout** (default 300s, max 3600s) | Read endpoints are O(1); keep any EE export/orchestration calls async or push them to a Cloud Run **Job** so the request returns fast. |
| **Cold starts** | Set `--min-instances=1` for the public API to keep one warm instance. The image is slim (`[serving]` extra only, multi-stage build) so starts are fast. |
| **Memory / CPU sizing** | `--memory=512Mi --cpu=1` is plenty for the serving API (pure-wheel stack). The TiTiler/`[geo]` variant needs more (`--memory=2Gi`). |
| **Image size / build** | Multi-stage `python:3.11-slim`, non-root user, `.dockerignore` keeps the build context to a few hundred KB (the 975 MB `.venv`, `.git`, caches, `outputs/`, `tests/` are excluded). |
| **Concurrency model** | Cloud Run scales by *instances*; we run **1 Uvicorn worker** per instance (swap to gunicorn+`WEB_CONCURRENCY` only if you raise per-instance concurrency). |

---

## 3. Prerequisites

```bash
# Authenticate and pick the project.
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
export PROJECT_ID="$(gcloud config get-value project)"
export REGION="asia-south1"            # Mumbai — closest to Indian command areas

# Enable the APIs we use.
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  redis.googleapis.com \
  vpcaccess.googleapis.com \
  earthengine.googleapis.com
```

---

## 4. Step-by-step: deploy the serving API to Cloud Run

### Option A — one command from source (Buildpacks/Cloud Build builds the image)

`gcloud run deploy --source .` will use `infra/Dockerfile` when present at the
repo root path it builds from. Point it at the Dockerfile explicitly to be safe:

```bash
gcloud run deploy agristress-api \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --min-instances=1 \
  --max-instances=10 \
  --memory=512Mi \
  --cpu=1 \
  --port=8080 \
  --set-env-vars=AGRISTRESS_CORS_ORIGINS="https://your-dashboard.example"
```

> If `--source .` does not pick up `infra/Dockerfile` automatically in your
> gcloud version, use **Option B** (explicit build) or copy/symlink the
> Dockerfile to the repo root.

### Option B — build & push to Artifact Registry, then deploy (recommended for CI)

```bash
# 1) Create an Artifact Registry repo (once).
gcloud artifacts repositories create agristress \
  --repository-format=docker --location="$REGION" \
  --description="AgriStress container images"

export IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/agristress/agristress-api:$(git rev-parse --short HEAD)"

# 2) Build with Cloud Build using our Dockerfile (no local Docker needed).
gcloud builds submit --tag "$IMAGE" --gcs-log-dir="gs://${PROJECT_ID}_cloudbuild/logs" .
#   …or locally:  docker build -f infra/Dockerfile -t "$IMAGE" . && docker push "$IMAGE"

# 3) Deploy.
gcloud run deploy agristress-api \
  --image "$IMAGE" \
  --region "$REGION" \
  --allow-unauthenticated \
  --min-instances=1 --max-instances=10 \
  --memory=512Mi --cpu=1 --port=8080
```

### Wiring Memorystore (Redis) + secrets + Earth Engine

```bash
# Memorystore instance (private IP) + Serverless VPC connector so Cloud Run
# can reach it (Cloud Run has no VPC by default).
gcloud redis instances create agristress-cache \
  --region="$REGION" --size=1 --redis-version=redis_7_0
REDIS_HOST="$(gcloud redis instances describe agristress-cache --region="$REGION" --format='value(host)')"

gcloud compute networks vpc-access connectors create agristress-vpc \
  --region="$REGION" --range=10.8.0.0/28

# Store the Earth Engine service-account key in Secret Manager (never in the image).
gcloud secrets create gee-sa-key --data-file=/path/to/gee-service-account.json

# Redeploy with state + secrets + EE config bound in.
gcloud run deploy agristress-api \
  --image "$IMAGE" \
  --region "$REGION" \
  --allow-unauthenticated \
  --min-instances=1 --max-instances=10 --memory=512Mi --cpu=1 --port=8080 \
  --vpc-connector=agristress-vpc \
  --set-env-vars=REDIS_URL="redis://${REDIS_HOST}:6379/0" \
  --set-env-vars=EE_PROJECT="${PROJECT_ID}" \
  --set-env-vars=GEE_SERVICE_ACCOUNT="ee-runner@${PROJECT_ID}.iam.gserviceaccount.com" \
  --set-env-vars=GEE_SERVICE_ACCOUNT_KEY="/secrets/gee/key.json" \
  --set-secrets=/secrets/gee/key.json=gee-sa-key:latest
```

Environment variables the app/CLI actually read:

| Variable | Read by | Effect |
|---|---|---|
| `PORT` | `infra/Dockerfile` CMD, `agristress serve` | Listen port (Cloud Run sets it; default `8080`). |
| `HOST` | `agristress serve` | Bind host (default `0.0.0.0`). |
| `REDIS_URL` | `agristress.serving.cache.Cache` | Use Redis/Memorystore; unset → in-memory LRU fallback. |
| `AGRISTRESS_CORS_ORIGINS` | `agristress.serving.api.create_app` | Comma-separated allowed CORS origins (default `*`). |
| `EE_PROJECT` | `gee/00_auth.py:init_ee` | GCP project for Earth Engine billing/quota. |
| `GEE_SERVICE_ACCOUNT` / `GEE_SERVICE_ACCOUNT_KEY` | `gee/00_auth.py:init_ee` | Headless EE auth (service account email + key path). |
| `EE_HIGH_VOLUME` | `gee/00_auth.py:init_ee` | Route EE through the high-volume endpoint for tile/batch. |

Verify:

```bash
URL="$(gcloud run services describe agristress-api --region="$REGION" --format='value(status.url)')"
curl -fsS "$URL/health"      # -> {"status":"ok","cache_backend":"redis",...}
curl -fsS "$URL/aoi"
```

---

## 5. Cloud Run **Job** for the batch pipeline

The pipeline stages (`ingest/fuse/features/train/advisory`) are run-to-completion
work, so deploy the **same image** as a Cloud Run Job with a different command:

```bash
gcloud run jobs create agristress-pipeline \
  --image "$IMAGE" \
  --region "$REGION" \
  --vpc-connector=agristress-vpc \
  --set-env-vars=EE_PROJECT="${PROJECT_ID}",GEE_SERVICE_ACCOUNT_KEY="/secrets/gee/key.json" \
  --set-secrets=/secrets/gee/key.json=gee-sa-key:latest \
  --memory=2Gi --cpu=2 --task-timeout=3600 \
  --command=agristress \
  --args=demo,--season,kharif        # or: --args=train,--aoi,CMD-001

# Run it now, or on a schedule.
gcloud run jobs execute agristress-pipeline --region "$REGION"
gcloud scheduler jobs create http agristress-nightly \
  --schedule="0 19 * * *" --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/agristress-pipeline:run" \
  --http-method=POST --oauth-service-account-email="run-invoker@${PROJECT_ID}.iam.gserviceaccount.com"
```

> For large fusion/training that needs many CPUs or a GPU, run the same image on
> **Cloud Batch** or a **Vertex AI Custom Job** instead — both accept a container
> image + command, so no code change is required.

---

## 6. Earth Engine auth on Cloud Run

The repo already implements the headless flow in `gee/00_auth.py` (`init_ee`):
it prefers a **service account** (`ee.ServiceAccountCredentials(...)` →
`ee.Initialize(credentials, project=...)`) and falls back to stored user creds.

1. Create an EE-registered service account and grant it the *Earth Engine
   Resource Viewer/Writer* role; register it at <https://signup.earthengine.google.com>.
2. Store its JSON key in **Secret Manager** (`gee-sa-key` above).
3. Mount it into the Cloud Run service/job and point the env vars at it:
   `GEE_SERVICE_ACCOUNT=...`, `GEE_SERVICE_ACCOUNT_KEY=/secrets/gee/key.json`,
   `EE_PROJECT=$PROJECT_ID`.
4. In code, initialise once at startup:

   ```python
   from importlib import import_module
   auth = import_module("gee._auth")     # importable shim for gee/00_auth.py
   auth.init_ee()                         # reads EE_PROJECT / GEE_SERVICE_ACCOUNT* env
   ```

The orchestration service never ships raw imagery — it submits `ee.*` graphs and
exports the gold products to GCS, which the serving API/dashboard then read.

---

## 7. Dashboard hosting options

The dashboard in `dashboard/` is plain static files (MapLibre, vanilla JS, no
build step). Three options, in order of operational simplicity:

1. **Same Cloud Run container** (default) — the image already copies `dashboard/`;
   mount it as static files from the API (one origin → no CORS, simplest auth).
2. **GCS bucket + Cloud CDN** — `gsutil rsync dashboard/ gs://your-dashboard`;
   cheapest, globally cached, fully decoupled from the API. Point `config.js` at
   the Cloud Run API URL and set `AGRISTRESS_CORS_ORIGINS` to the dashboard origin.
3. **Firebase Hosting** — `firebase deploy` for a managed CDN + custom domain +
   easy preview channels.

---

## 8. When you DO need a VM / Vertex GPU

Reach for a VM or a Vertex AI GPU job **only** in these cases — none are required
for the PS6 deliverables:

- **Persistent deep-learning fusion / foundation-model fine-tuning** (the `[dl]`
  extra: torch/timm spatio-temporal backbones). Use a **Vertex AI Custom Job**
  (preferred — managed, scale-to-zero) or a GPU VM if you need a long-lived,
  interactive training box.
- **Large, continuous local GDAL/raster processing** that you deliberately keep
  *off* Earth Engine (e.g. bulk reprojection of proprietary scenes). A Cloud Run
  Job or Cloud Batch usually still suffices; a VM only helps when you need a
  persistent mounted disk and a steady-state worker.
- **Stateful services that cannot scale to zero.** AgriStress has none — the
  cache is Memorystore, products are GCS/BigQuery.

Everything else (serving, EE orchestration, batch, tiles) is serverless.

---

## 9. Cost / scaling & security notes

**Cost / scaling.** Cloud Run **scales to zero** — you pay per request-second, so
the public API with `--min-instances=1` costs only one always-warm small instance
plus request load. Earth Engine compute is billed to `EE_PROJECT`. Memorystore
(1 GB) and a VPC connector are the main fixed costs; drop them and rely on the
in-memory LRU if you don't need a shared cache. PMTiles-on-CDN tiles are
effectively free at the edge versus a running TiTiler service.

**Security.**
- **Secrets**: EE keys and Redis auth live in **Secret Manager**, mounted at
  runtime — never in the image, env file, or git (`.gitignore`/`.dockerignore`
  already exclude `service-account*.json`, `*.key`, `*.pem`, `.env`).
- **Least-privilege service account**: give the Cloud Run runtime SA only the
  roles it needs (Secret Manager accessor, the EE role, GCS object access on the
  product bucket). Don't reuse the default Compute SA.
- **CORS**: set `AGRISTRESS_CORS_ORIGINS` to the real dashboard origin in
  production instead of the permissive `*` default.
- **Auth**: `--allow-unauthenticated` is fine for a public read API; for an
  internal tool put **IAP** / `--no-allow-unauthenticated` in front and invoke
  with an identity token.

---

## 10. Local parity (compose) & quick reference

```bash
# Local stack mirroring the Cloud Run topology (API + Redis).
docker compose -f infra/docker-compose.yml up --build
curl localhost:8080/health        # cache_backend should be "redis"

# Run the same image standalone on a custom port (proves $PORT binding).
docker build -f infra/Dockerfile -t agristress-api .
docker run --rm -e PORT=9000 -p 9000:9000 agristress-api
curl localhost:9000/health

# No Docker? The app honours $PORT directly:
PORT=8099 agristress serve        # binds 0.0.0.0:8099
# or: PORT=8099 python -m uvicorn agristress.serving.api:app --host 0.0.0.0 --port 8099

# Deploy shortcut:
make deploy-cloudrun PROJECT_ID=your-project REGION=asia-south1
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness + cache backend + cell/AOI counts (used by the healthcheck). |
| `GET /crop\|/stress\|/advisory?h3=&date=` | O(1) keyed lookups. |
| `GET /timeseries?h3=&var=&start=&end=` | Per-variable series. |
| `GET /tiles/{layer}/{z}/{x}/{y}.png` | Demo PNG tiles (`/cog/...` when TiTiler/`[geo]` is installed). |
| `GET /aoi`, `GET /command/{id}/rollup` | Command-area metadata + advisory rollup. |

See also: [`docs/PLATFORM_O1.md`](./PLATFORM_O1.md) (serving/H3 design),
[`docs/ARCHITECTURE.md`](./ARCHITECTURE.md) (system), [`gee/README.md`](../gee/README.md)
(Earth Engine scripts).
