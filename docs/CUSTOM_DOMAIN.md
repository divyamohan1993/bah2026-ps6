# Custom Domain — `agristress.dmj.one` (Zero-Cost)

This maps a public custom domain to the AgriStress Cloud Run service **without
adding any fixed cost** and **without breaking scale-to-zero**.

- Script: [`scripts/map_domain.sh`](../scripts/map_domain.sh)
- Default domain: `agristress.dmj.one` (override with `DOMAIN=...`)
- Service: `agristress-api` · Region: `asia-east1` · Project: `dmjone`

---

## 1. Why a Cloud Run domain mapping (the zero-cost choice)

| Option | Fixed cost | Scale-to-zero? | Wildcard `*.dmj.one`? | Verdict |
|---|---|---|---|---|
| **Cloud Run domain mapping** (this) | **$0** | **Yes** | No (one subdomain) | **Chosen** — free, keeps min=0, perfect for a single subdomain |
| Global external HTTPS Load Balancer | ~**$18+/mo** (forwarding rule, always-on) | Yes (backend) | Yes | Needed only for a true wildcard; **breaks zero-cost** |
| Firebase Hosting (rewrite to Cloud Run) | $0 on Spark (free) tier | Yes | Per-site | Viable free alternative; adds a Firebase project + `firebase.json` |

We need exactly **one** subdomain (`agristress.dmj.one`), so a **Cloud Run
domain mapping** is the right tool: it is **free**, supported in `asia-east1`,
provisions a **Google-managed SSL certificate** automatically, and leaves the
service **scale-to-zero** and **`--allow-unauthenticated`** behind it.

A **global external Load Balancer** would only be justified if you needed a
**wildcard** (`*.dmj.one`) or multi-region anycast — but its forwarding rule is
an always-on resource (~$18/mo), which **breaks the zero-cost goal**, so it is
deliberately *not* used here. **Firebase Hosting** is a reasonable free
alternative (rewrite a Hosting site to the Cloud Run service) if you later want
a CDN edge or already use Firebase.

---

## 2. Steps

### (1) Verify domain ownership — one-time, manual

Cloud Run will only map a domain you have **verified ownership** of. In Cloud
Shell:

```bash
gcloud domains verify dmj.one
```

This opens **Google Search Console / Webmaster Central**, which gives you a
**TXT** record to add at the `dmj.one` registrar. Add it, confirm verification,
then continue. (This is registrar-side and cannot be automated from here.)

### (2) Create the mapping

```bash
bash scripts/map_domain.sh
# or a different subdomain:  DOMAIN=agristress.example.org bash scripts/map_domain.sh
```

The script is idempotent (describe-or-create). It creates the mapping with
`gcloud beta run domain-mappings create` and then prints the DNS record(s) you
must add.

### (3) Add the printed DNS record at the registrar

For a **subdomain** like `agristress.dmj.one`, Cloud Run returns a **CNAME**:

```
agristress   CNAME   ghs.googlehosted.com.
```

Add that at the `dmj.one` DNS provider. (If you ever map an **apex/root** domain
instead, Cloud Run returns **A/AAAA** records rather than a CNAME — add exactly
what the script prints.)

### (4) Wait for Google-managed SSL

Once DNS resolves, Google **auto-issues a managed TLS certificate** — typically
a few minutes, occasionally up to ~24h. Then:

```bash
curl -fsS https://agristress.dmj.one/health
```

---

## 3. What stays the same

- The Cloud Run service remains **`--allow-unauthenticated`** (public read API)
  and **scale-to-zero** (`--min-instances=0 --max-instances=1`) behind the
  domain. The mapping itself has **no fixed cost**.
- HTTP is auto-redirected to HTTPS by Cloud Run; the managed cert auto-renews.

> **Honest note:** steps (1) ownership verification and (3) adding DNS records
> are **manual, registrar-side** actions on `dmj.one` — they cannot be performed
> from this repo. The script automates the Cloud Run side (mapping + surfacing
> the exact records to add).

See also [`docs/DEPLOYMENT.md`](./DEPLOYMENT.md) and
[`docs/CICD.md`](./CICD.md).
