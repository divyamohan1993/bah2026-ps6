# Irrigation Advisory Methodology — Satellite + Weather → 8-Day Crop-Water-Deficit for Canal Command Areas

**ISRO Bharatiya Antariksh Hackathon (BAH) 2026 — Problem Statement 6 (PS6)**
*Deep, quantitative agro-hydrology methodology for an operational 8-day crop-water-deficit and irrigation advisory over canal command areas in India.*

---

## 0. Purpose, Scope, and Design Philosophy

The objective is to convert **freely available satellite imagery + gridded weather** into a **physically-defensible, 8-day-ahead crop-water-deficit (CWD) and irrigation advisory** for canal command areas, resolved from the **pixel → field → outlet/chak → distributary** scale, and expressed in the operational language farmers and canal engineers actually use: *when to irrigate, and how many mm (and how many m³ at the outlet)*.

Three independent information streams are fused so that **no single sensor failure or cloud gap breaks the advisory**:

| Stream | Question it answers | Primary product |
|---|---|---|
| **(A) Atmospheric demand** | How much water *wants* to leave the crop? | FAO-56 Penman-Monteith **ET₀** from gridded weather |
| **(B) Crop demand & state** | How much does *this crop, at this stage* transpire? | **Kc·ET₀ = ETc**, with Kc constrained by **NDVI** |
| **(C) Surface state / actual ET / stress** | What is the soil-plant system *actually* doing? | Thermal **ETa** (energy balance) + **SMAP/ASCAT/EOS-04 soil moisture** |

These feed a **FAO-56 Chapter-8 root-zone soil-water balance** that is the heart of the system. The 8-day deficit is simply the **root-zone depletion `Dr`** projected to the end of an 8-day window. Satellite ETa and soil moisture are used to **initialize, validate, and data-assimilate** (nudge / EnKF) the water balance so model drift is bounded by observations.

> **Why 8 days?** It matches (i) the **warabandi** canal rotation cycle (typically 7–8 days; ~1 h of flow per acre per week), (ii) Landsat/Sentinel-2 revisit and the **8-day MODIS/MOD16 compositing cadence**, and (iii) the practical lead-time of a reliable medium-range forecast (IMD-GFS). See §7.3.

**Golden rule (publishing gate, §8):** an advisory is only published for a command area when **the demand model (ETc), the energy-balance observation (ETa), the soil-moisture observation (θ), and the stress indices (VCI/TVDI/CWSI) tell one coherent story.** Disagreement → flag, don't advise.

---

## 1. Atmospheric Demand — FAO-56 Penman-Monteith Reference ET₀

### 1.1 The full standardized daily equation (Allen et al., 1998; ASCE-EWRI 2005)

For the **standardized short (grass) reference**, daily step:

```
        0.408·Δ·(Rn − G) + γ·(Cn/(T + 273))·u2·(es − ea)
ET0 = ─────────────────────────────────────────────────────
                  Δ + γ·(1 + Cd·u2)
```

with **daily grass-reference constants**:

| Symbol | Daily grass (ETo) | Hourly | Meaning |
|---|---|---|---|
| `Cn` | **900** | 37 | numerator constant [K·mm·s³·Mg⁻¹·d⁻¹] |
| `Cd` | **0.34** | 0.24 (day) / 0.96 (night) | denominator wind constant [s·m⁻¹] |

- `ET0` — reference evapotranspiration [mm d⁻¹]
- `Rn` — net radiation at crop surface [MJ m⁻² d⁻¹]
- `G` — soil heat flux [MJ m⁻² d⁻¹]; **daily G ≈ 0** (negligible over a day)
- `T` — mean daily air temp at 2 m [°C], `T = (Tmax + Tmin)/2`
- `u2` — wind speed at 2 m [m s⁻¹]
- `es` — saturation vapour pressure [kPa]; `ea` — actual vapour pressure [kPa]
- `Δ` — slope of saturation vapour-pressure curve [kPa °C⁻¹]
- `γ` — psychrometric constant [kPa °C⁻¹]

### 1.2 All intermediate quantities (compute in this order)

**(a) Slope of the saturation vapour-pressure curve, Δ**
```
        4098 · [0.6108 · exp(17.27·T / (T + 237.3))]
Δ = ─────────────────────────────────────────────────     [kPa °C⁻¹]
                       (T + 237.3)²
```

**(b) Saturation vapour pressure at temperature T, e°(T)** (Tetens)
```
e°(T) = 0.6108 · exp( 17.27·T / (T + 237.3) )              [kPa]
```

**(c) Mean saturation vapour pressure, es** — use Tmax & Tmin (do NOT use e°(Tmean), it underestimates):
```
es = [ e°(Tmax) + e°(Tmin) ] / 2                            [kPa]
```

**(d) Actual vapour pressure, ea** — pick the best available:
```
from RHmax/RHmin :  ea = [ e°(Tmin)·RHmax/100 + e°(Tmax)·RHmin/100 ] / 2
from RHmean      :  ea = (RHmean/100) · es
from dewpoint Td :  ea = e°(Td) = 0.6108·exp(17.27·Td/(Td+237.3))
humid-data-scarce:  ea ≈ e°(Tmin)   (assumes Tmin ≈ dewpoint)
```

**(e) Atmospheric pressure from elevation z [m]** (simplified ideal-gas / lapse):
```
P = 101.3 · [ (293 − 0.0065·z) / 293 ] ^ 5.26              [kPa]
```

**(f) Psychrometric constant**
```
γ = (cp · P) / (ε · λ) = 0.000665 · P                       [kPa °C⁻¹]
```
(cp = 1.013×10⁻³ MJ kg⁻¹ °C⁻¹, λ ≈ 2.45 MJ kg⁻¹, ε = 0.622.)

### 1.3 Net radiation, Rn = Rns − Rnl

**Extraterrestrial radiation `Ra`** [MJ m⁻² d⁻¹] (J = day of year, φ = latitude in rad):
```
dr = 1 + 0.033·cos(2π·J/365)                         (inverse rel. Earth-Sun dist.)
δ  = 0.409·sin(2π·J/365 − 1.39)                       (solar declination, rad)
ωs = arccos(−tan(φ)·tan(δ))                           (sunset hour angle, rad)
Ra = (24·60/π)·Gsc·dr·[ ωs·sin(φ)·sin(δ) + cos(φ)·cos(δ)·sin(ωs) ]
Gsc = 0.0820 MJ m⁻² min⁻¹  (solar constant)
```
Clear-sky radiation `Rso = (0.75 + 2×10⁻⁵·z)·Ra`.
Solar radiation from sunshine hours (if `Rs` not measured): `Rs = (as + bs·n/N)·Ra`, `as=0.25, bs=0.50`, `N = (24/π)·ωs`. **Preferably use satellite insolation `Rs` directly (INSAT-3D / Kalpana).**

**Net shortwave** (albedo α = 0.23 for grass reference):
```
Rns = (1 − α)·Rs = 0.77 · Rs                          [MJ m⁻² d⁻¹]
```

**Net longwave (Stefan-Boltzmann form)**:
```
        ⎡ Tmax,K⁴ + Tmin,K⁴ ⎤
Rnl = σ·⎢ ───────────────── ⎥·(0.34 − 0.14·√ea)·(1.35·Rs/Rso − 0.35)
        ⎣        2          ⎦
σ = 4.903×10⁻⁹  MJ K⁻⁴ m⁻² d⁻¹   ;  Tmax,K = Tmax+273.16, etc.
```
- `(0.34 − 0.14·√ea)` — net emissivity (humidity) term
- `(1.35·Rs/Rso − 0.35)` — cloudiness factor (cap `Rs/Rso ≤ 1`)

```
Rn = Rns − Rnl
```

### 1.4 Weather-grid source table (rank for India)

| Product | Native res. | Step | Vars for ET₀ | Notes / use |
|---|---|---|---|---|
| **AgERA5 (C3S Agromet)** | **~0.1° (~10 km)** | daily | Tmax/Tmin, RH/Td, u, Rs | **Best general default**; bias-corrected, agromet-ready, global, daily-complete |
| ERA5-Land | ~9 km | hourly→daily | all | Long record (1950–), good for climatology/Kc-mid adjust |
| **NASA POWER** | ~0.5° | daily | all, incl. Rs | Easy API, validated for ET₀, gap-free; coarse |
| GLDAS (Noah) | 0.25° | 3-hourly | all + soil | Useful for G, soil-moisture priors |
| **IMD gridded** | **0.25° rain / 1° temp** | daily | Tmax/Tmin, rain | **National benchmark** for T and *especially rainfall* |
| **INSAT-3D / Kalpana** | ~4–8 km | hourly/daily | **Rs (insolation)**, LST | Geostationary; drives `Rns` and ALEXI morning LST rise |

### 1.5 The India-ready ~12.5 km daily ET₀ grid (operational target)

Fuse **IMD-GFS short-range forecast (~12.5 km)** meteorology (Tmax/Tmin, RH, u2) with **INSAT-3D / Kalpana insolation (Rs)** through the FAO-56 PM engine above to produce a **daily ~12.5 km gridded ET₀** over the Indian landmass (cf. MOSDAC/SAC near-real-time ET₀ products: validated daily R² ≈ 0.34–0.90, MAPE 10–27% vs station ET₀; monthly ET₀ ranges ~10–350 mm Jan→May). This is the **demand backbone** for the advisory and is forecast-capable out to the 8-day window. *(Allen 1998; Bhattacharya/SAC; Springer ESI 2023, see refs.)*

---

## 2. Crop Water Demand — Crop Coefficients (Kc) and the NDVI Link

### 2.1 Single vs dual coefficient

```
Single:   ETc      = Kc · ET0
Dual:     ETc      = (Kcb + Ke) · ET0           ← PREFERRED (separates T from soil E)
Constraint:   Kcb + Ke ≤ Kc,max ≈ 1.2 … 1.4
```
- `Kcb` — **basal** crop coefficient (transpiration + diffusive soil E at dry surface)
- `Ke` — **soil-evaporation** coefficient (spikes after rain/irrigation, decays as topsoil dries)
- `Kc,max` ≈ `max{ 1.2 + [0.04(u2−2) − 0.004(RHmin−45)]·(h/3)^0.3 , Kcb+0.05 }`

The dual method is preferred because it (a) models the post-wetting evaporation spike explicitly (critical for the 8-day balance after a canal turn or rain), and (b) cleanly couples to the soil-water balance via `Ke = Kr·(Kc,max − Kcb)` where `Kr` is the topsoil evaporation-reduction coefficient.

### 2.2 Stage-wise Kc table (FAO-56 Table 11/12; 4-stage curve)

The Kc curve has **four growth stages** — *initial (L_ini), crop-development (L_dev), mid-season (L_mid), late-season (L_late)* — and a **trapezoidal shape**: flat at `Kc_ini`, linear rise through development to `Kc_mid`, flat at `Kc_mid`, linear fall through late season to `Kc_end` (at harvest/maturity).

| Crop | L_ini | L_dev | L_mid | L_late | **Kc_ini** | **Kc_mid** | **Kc_end** | Notes |
|---|---|---|---|---|---|---|---|---|
| **Rice / Paddy** | 30 | 30 | 60 | 30 | **1.05** | **1.20** | **0.90→0.60** | flooded; +percolation term (§2.6); 1.20 ↔ ponded |
| **Wheat (winter)** | 30 | 140 | 40 | 30 | **0.30** | **1.15** | **0.25 (→0.40)** | 0.25 if dry/dead, 0.40 if green at harvest |
| **Maize (grain)** | 30 | 40 | 50 | 30 | **0.30** | **1.20** | **0.60 (→0.35)** | 0.35 for dry grain |
| **Cotton** | 30 | 50 | 60 | 55 | **0.35** | **1.15–1.20** | **0.70→0.50** | |
| **Sugarcane** | 35 | 60 | 190 | 120 | **0.40** | **1.25** | **0.75** | long mid-season |
| **Pulses (legumes)** | 20 | 30 | 35 | 20 | **0.40** | **1.15** | **0.35** | e.g. chickpea/gram |

*(Stage lengths are indicative for a typical season; calibrate to local sowing dates / GDD — see §2.5.)*

### 2.3 Climate adjustment of Kc_mid and Kc_end

For `Kc_mid` or `Kc_end ≥ 0.45`, adjust tabulated values from the FAO standard climate (RHmin=45%, u2=2 m/s) to local conditions:
```
Kc = Kc(Tab) + [ 0.04·(u2 − 2) − 0.004·(RHmin − 45) ] · (h/3)^0.3
```
`h` = mean plant height [m] during the stage. Arid/windy India (low RHmin, high u2) **raises** Kc_mid by 0.05–0.10.

### 2.4 Kc–NDVI empirical link (makes Kc spatial & real-time)

Tabulated Kc is a calendar template; **NDVI from Sentinel-2 / Landsat / Resourcesat / MODIS makes it pixel-specific and responsive to actual canopy development, gaps, lodging, stress**:
```
Kc = 1.457 · NDVI − 0.1725            (r² ≈ 0.90; many crops, e.g. Kamble et al. 2013)
```
Use this to **override/anchor the calendar Kc**, especially for the development and senescence limbs where calendar timing is least reliable.

### 2.5 Fractional-cover / SIMS route (physically cleaner for Kcb)

The **SIMS** (Satellite Irrigation Management Support) approach derives `Kcb` from green canopy fraction:
```
fc   = (NDVI − NDVI_min) / (NDVI_max − NDVI_min)          (fraction green cover, 0–1)
                                                          NDVI_min≈0.15(bare), NDVI_max≈0.90(full)
Kd   = min( 1, ML·fc_eff,  fc_eff^(1/(1+h)) )             (density coefficient)
Kcb  = Kcb,min + Kd · (Kcb,full − Kcb,min)
       Kcb,full ≈ min(1.0 + 0.1·h, 1.20) + climate-adjust
```
This is the basis of OpenET-SIMS and is the recommended NDVI→Kcb path when canopy height/cover data exist; the simple linear `Kc=1.457·NDVI−0.1725` is the lightweight fallback.

### 2.6 Paddy / puddled-rice special term

Flooded rice is **not** a simple ETc crop. The field water balance must add **deep percolation + seepage** through the puddled hardpan:
```
ETc(rice) = Kc·ET0          (Kc≈1.05→1.20→0.60, often +0.10–0.15 vs upland due to ponded water)
Field demand = ETc + Percolation(2–6 mm/d typ., up to 15+ on light soils) + maintain ponding depth
```
For **AWD (Alternate Wetting & Drying)** the advisory switches from "maintain 5 cm ponding" to "re-flood when perched water table drops ~15 cm below surface" (§7.4) — a major water saver.

---

## 3. Satellite Actual ET (ETa) — Surface Energy Balance (independent check & assimilation source)

The land-surface **energy balance** partitions net radiation into sensible (`H`) and latent (`λE`) heat:
```
λE = Rn − G − H          [W m⁻²]      →    ETa = λE / λ   [mm]
```
`λE` (latent heat flux) is the residual; the models below differ mainly in **how they estimate H from land-surface temperature (LST)**. ETa is our **independent observation of what the crop is actually doing**, used to validate `Ks` and to assimilate into the water balance (§6).

### 3.1 Model family (transferable to India; all run on GEE / EOS-04 / Landsat)

| Model | Core idea / H-estimation | Anchoring | Pros / cons for India |
|---|---|---|---|
| **SEBAL** (Bastiaanssen) | `dT = a + b·Ts` calibrated from **hot** (dry, λE=0) & **cold** (wet, H≈0) anchor pixels in the scene | self-calibrating, no in-situ data | Robust; anchor-pixel selection is subjective/automatable |
| **METRIC / eeMETRIC** | SEBAL + anchors tied to **alfalfa reference ETr** at cold pixel | ETr-anchored | Reduces bias; needs good weather grid |
| **SSEBop** (Senay) | **"Satellite psychrometry":** `ETf = (Th − Ts)/dT`, then `ETa = ETf·k·ET0`; `dT` is a **predefined seasonally/spatially dynamic** clear-sky temp difference; `Th` = hot/dry-bulb limit | reference-ET anchored, *no in-scene hot/cold pixel search* | **Most operational / cloud-gap-tolerant; recommended primary ETa for an operational pipeline** |
| **SEBS** (Su) | Bulk atmospheric similarity; evaporative fraction from dry/wet limits | physical limits | Needs more met inputs |
| **PT-JPL** | **Priestley-Taylor** `λE = α·f·(Δ/(Δ+γ))·(Rn−G)`; ecophysiological constraints **f** from VI/RH/VPD — **no LST in classic form** | radiation-driven | Cloud-robust (optical only); weak under advection |
| **ALEXI / DisALEXI** | Two-source land-atmosphere model driven by **GOES/INSAT morning LST rise**; DisALEXI sharpens to **Landsat 30 m** | time-differential LST | Geostationary-friendly (INSAT-3D!), 30 m disaggregation |

`ETf` in SSEBop is the **ET fraction** (≈ evaporative fraction), nominally 0–1, multiplied by user reference ET (`k`≈1.0–1.25 max-ET scalar) — exactly the bridge that lets a 30–100 m LST snapshot be converted to a daily/8-day ETa using the same ET₀ grid from §1.

### 3.2 OpenET ensemble (the validation gold standard, methods transferable)

**OpenET** runs **six models on Google Earth Engine at 30 m**: **DisALEXI, eeMETRIC, geeSEBAL, PT-JPL, SIMS, SSEBop**, combined by a **MAD (median-absolute-deviation) ensemble** (robust median, outlier models down-weighted).

> **Accuracy (Melton et al., *Nature Water* 2023; 152 flux/EC stations):** ensemble **r² ≈ 0.90** at cropland; cropland MAE **≈ 15.8 mm/month (~17% of mean)**, MBE ≈ −5.3 mm/month; **per-model 0.74–1.07 mm/day** at daily step; the **ensemble agreement / spread is ~0.2–0.3 mm/day** for well-behaved crop pixels. ⇒ At field scale over irrigated crops, satellite ETa is now accurate enough to **drive water accounting**, not merely qualitative monitoring.

For PS6, build the **same multi-model ensemble over India** (Landsat 8/9 + Sentinel-2 + EOS-04/INSAT LST), and use the **ensemble ETa as the independent stress observation** to (i) validate the modeled `Ks` (§4), (ii) catch un-modeled stress (disease, salinity, mis-timed irrigation), and (iii) assimilate (§6).

---

## 4. Root-Zone Soil-Water Balance (FAO-56 Chapter 8) — the engine

### 4.1 Depletion form (run daily, report at 8 days)

Track **root-zone depletion `Dr`** [mm] below field capacity:
```
Dr,i = Dr,i−1 − (P − RO)i − Ii − CRi + ETc,adj,i + DPi
```
| Term | Meaning |
|---|---|
| `Dr,i` | root-zone depletion at end of day *i* [mm]; `0 ≤ Dr ≤ TAW` |
| `P` | precipitation [mm]; `RO` runoff (SCS-CN, §4.5) |
| `I` | net irrigation reaching root zone [mm] |
| `CR` | capillary rise from a shallow water table [mm] (≈0 if WT deep) |
| `ETc,adj` | **stress-adjusted** crop ET [mm] (= `Ks·(Kcb+Ke)·ET0`, §4.4) |
| `DP` | deep percolation below root zone [mm] (`DP = max(0, (P−RO)+I−ETc,adj−Dr,i−1)`) |

**Storage (mass-balance) equivalent** (useful for assimilation/diagnostics):
```
ΔSW = P + I − ETa − DP − RO            (SW = root-zone water storage [mm])
```

### 4.2 Total & Readily Available Water

```
TAW = 1000 · (θFC − θWP) · Zr          [mm]   (Zr = root depth [m])
RAW = p · TAW                          [mm]
```
- `θFC` = field capacity, `θWP` = wilting point [m³ m⁻³]
- `Zr` = effective rooting depth — grows with crop stage (cap at crop max)
- `p` = fraction of TAW depletable **before stress** (transpiration drops below potential)

### 4.3 Depletion fraction `p` (and its ET-rate correction)

Default `p ≈ 0.55`; tabulated range **0.30–0.70** by crop. Adjust for evaporative demand:
```
p = p_Tab + 0.04·(5 − ETc)             (limit 0.1 ≤ p ≤ 0.8; ETc in mm/d)
```
High ETc days ⇒ smaller `p` ⇒ stress onset earlier. **Tighten `p` (e.g. 0.40–0.50) at sensitive stages** (flowering, grain-fill) — see §7.2.

### 4.4 Water-stress coefficient `Ks` and stress-adjusted ETc

```
        TAW − Dr          TAW − Dr
Ks =  ───────────  =  ───────────────       for Dr > RAW
        TAW − RAW        (1 − p)·TAW

Ks = 1                                       for Dr ≤ RAW (no stress)

ETc,adj = Ks · (Kcb + Ke) · ET0   =   Ks · Kc · ET0
```
`Ks` falls linearly from 1 (at `Dr=RAW`) to 0 (at `Dr=TAW`). It is the **single most important advisory variable**: it converts "depletion" into "yield-relevant stress," and is the quantity we **cross-check against satellite ETa/CWSI/SM** before publishing.

### 4.5 Soil hydraulic properties θFC / θWP (texture → from SoilGrids / NBSS-LUP)

Derive per-pixel `θFC, θWP` from **SoilGrids 250 m** or **NBSS&LUP** texture + organic-carbon via **pedotransfer functions (Saxton-Rawls 2006)**. Indicative values:

| Texture | θFC (m³/m³) | θWP (m³/m³) | AWC=θFC−θWP | TAW @ Zr=1 m |
|---|---|---|---|---|
| Sand | 0.10 | 0.05 | 0.05 | 50 mm |
| **Sandy loam** | **0.21** | **0.09** | **0.12** | **120 mm** |
| Loam | 0.27 | 0.12 | 0.15 | 150 mm |
| Silt loam | 0.30 | 0.13 | 0.17 | 170 mm |
| Clay loam | 0.33 | 0.18 | 0.15 | 150 mm |
| Clay | 0.39 | 0.24 | 0.15 | 150 mm |

### 4.6 Effective rainfall `Peff`

Not all rain is useful (runoff + percolation lost). Two tiers:

**USDA-SCS monthly (quick):**
```
Peff = P·(125 − 0.2·P)/125     for P ≤ 250 mm/month
Peff = 125 + 0.1·P             for P > 250 mm/month
```

**Daily SCS Curve-Number (PREFERRED for the 8-day balance):**
```
S = (25400/CN) − 254                                    [mm]   (CN by soil group+cover+AMC)
RO = (P − 0.2S)² / (P + 0.8S)   if P > 0.2S, else 0      [mm]
Peff = P − RO                                            (then percolation handled by DP term)
```
Daily CN respects storm intensity and antecedent moisture (AMC adjusted from current `Dr`), which the monthly formula cannot — essential at 8-day resolution.

---

## 5. The 8-Day Crop-Water Deficit and Irrigation Depths

Integrate the daily balance over the 8-day window (today → +7 d, using forecast ET₀ & forecast/expected rain):

```
Deficit_8d  =  Σ ETc,adj  −  ( Σ Peff + ΔS_soil )  =  Dr (end of window)
```
i.e. the 8-day deficit **is** the projected end-of-window root-zone depletion. From it:

```
Net irrigation depth required:   dnet = Dr = (θFC − θ)·Zr·1000         [mm]
                                          (top-up from current θ to field capacity)
Gross (delivered) depth:         dgross = dnet / Ea                    [mm]
```

**Application/field efficiency `Ea`:**

| Method | Ea |
|---|---|
| Surface (border/furrow/basin) | **0.55–0.65** |
| Sprinkler | **0.75** |
| Drip / micro | **0.90** |

To **avoid stress**, irrigation is scheduled so that `Dr` does not exceed `RAW` (i.e. keep `Ks=1`); the *trigger* is `Dr → RAW`, the *dose* is `dgross` to refill to `θFC` (or to a target below FC to leave room for forecast rain).

---

## 6. Soil-Moisture Integration — Initialization, Assimilation, Validation

Modeled `Dr` drifts (errors in ET₀, Kc, rain, percolation accumulate). **Satellite soil moisture resets and constrains it.**

### 6.1 Soil-moisture product table

| Product | Sensor/principle | Native res. | Depth | Revisit | Use |
|---|---|---|---|---|---|
| **SMAP** (L2/L3/L4) | L-band radiometer | 9–36 km (L4 9 km RZSM) | ~5 cm (L4→RZ) | 2–3 d | Best passive SSM; **L4 already gives root-zone** |
| SMOS | L-band radiometer | ~40 km | ~5 cm | ~3 d | Long record |
| **ASCAT** (Metop) | C-band scatterometer | 12.5–25 km | ~2 cm | ~1–2 d | **Daily, all-weather; native SWI product** |
| Sentinel-1 | C-band SAR | **~1 km (down to 100 m)** | ~2–5 cm | 6–12 d | High-res field-scale SSM |
| **EOS-04 (RISAT-1A)** | C-band SAR (ISRO) | **~500 m–1 km** | ~5 cm | ~12 d | **India operational (Bhoonidhi/MOSDAC)** |
| **NISAR** (2024+) | L+S SAR (ISRO-NASA) | **~100 m** | ~5–10 cm | ~12 d | Game-changer for field-scale SM over India |

### 6.2 Surface → root-zone: exponential filter (Soil Water Index)

Satellites mostly sense the **top ~2–5 cm**. Propagate to the root zone with the **Wagner (1999) exponential filter / Soil Water Index**, recursive form:
```
                Σ_i  SSM(ti) · e^{ −(tn − ti)/T }
SWI(tn)  =  ──────────────────────────────────────
                   Σ_i  e^{ −(tn − ti)/T }

Recursive:   SWI(tn) = SWI(tn−1) + Kn · ( SSM(tn) − SWI(tn−1) )
             Kn = Kn−1 / ( Kn−1 + e^{ −(tn − tn−1)/T } )
```
- `T` = **characteristic time length** [days] — increases with root depth/soil; surface-near `T≈6 d`, deeper RZ `T≈14–43 d` (calibrate per texture/depth).
- `SWI ∈ [0,1]` is mapped to volumetric `θ` via `θ = θWP + SWI·(θFC − θWP)` (or scale to porosity), giving an **independent root-zone θ** to compare against the balance.

### 6.3 Assimilation strategy

1. **Initialize / reset** `Dr` from observed θ at each overpass:
   `Dr = (θFC − θ_obs)·Zr·1000` (hard reset when obs is fresh & cloud-free).
2. **Nudge** between overpasses: `Dr_corr = Dr_model + G·(Dr_obs − Dr_model)`, gain `G` from relative error variances.
3. **EnKF** (operational best): run an ensemble of the §4 balance, update state `Dr` (and optionally Kc, percolation) with SMAP/ASCAT/EOS-04 innovations; propagates uncertainty into the advisory confidence.
4. **Validate `Ks`**: when θ_obs implies `Dr>RAW`, confirm modeled `Ks<1` and that **satellite ETa shows the matching depression**. Three-way agreement (θ, Ks, ETa) is the §8 gate.

---

## 7. Advisory Logic — Status, Scheduling, Command-Area Aggregation

### 7.1 Five-class status (depletion vs RAW, gated by Ks & stage)

| Status | Condition (Dr vs RAW/TAW, Ks) | Meaning | Action |
|---|---|---|---|
| 🟢 **No-irrigation** | `Dr ≤ 0.5·RAW`, `Ks=1` | ample water | none; let crop draw down |
| 🔵 **Watch** | `0.5·RAW < Dr ≤ RAW`, `Ks=1` | approaching trigger | plan turn; monitor forecast |
| 🟡 **Irrigate-soon** | `Dr > RAW`, `Ks` just `< 1` | stress beginning | schedule within ~1–2 days |
| 🟠 **Irrigate-now** | `Ks ≲ 0.8` (`Dr` well past RAW) | yield-affecting stress | irrigate immediately, `dgross` |
| 🔴 **Critical** | `Dr → TAW`, `Ks → 0` | wilting / crop loss | emergency; prioritize at outlet |

Thresholds are **stage-aware**: at flowering/grain-fill use a **tighter `p`** so 🟡/🟠 trip earlier.

### 7.2 Stage-aware tightening

Apply smaller `p` (earlier irrigation trigger) during yield-critical stages:
`wheat` crown-root-initiation & flowering; `maize` tasseling–silking; `cotton` flowering–boll; `rice` panicle-initiation→flowering. Implement by reducing `p_Tab` by ~0.1 in those GDD windows (§2.5 phenology).

### 7.3 Scheduling output (per field/pixel)

```
WHEN:  date at which Dr is forecast to reach RAW (Ks about to drop below 1)
HOW MUCH:  dnet = Dr_at_trigger ;  dgross = dnet / Ea
           (optionally cap to leave headroom = forecast Σ Peff over next days)
```

### 7.4 Rice ponding / AWD logic (override)

- **Continuous flooding:** target ponding 5 cm; advise re-flood when ponding → 0; field demand = `ETc + percolation` (§2.6).
- **AWD (recommended water-saver):** allow drawdown to ~**−15 cm** perched water table after canopy closure (safe AWD), then re-flood; suspend AWD at flowering. Cuts seasonal water 15–30% with little/no yield loss.

### 7.5 Canal command aggregation (pixel → field → chak → distributary)

```
Aggregate fields served by an outlet (chak):
   V_chak = Σ_fields ( dgross,f · A_f )            [m³]     (A_f = field area)
   Turn (flow) time at the outlet:  t = V_chak / Q   [s]    (Q = outlet discharge, m³/s)
Roll up:  outlet → minor → distributary → main canal demand.
```
- **Warabandi link:** a fixed rotational roster (≈ **1 h/acre/week**, 7–8-day cycle) is *exactly* an 8-day allocation problem ⇒ the **8-day advisory cadence aligns with the canal's own clock**, so outputs are directly actionable as turn lengths.
- **Equity / tail-end priority:** rank chaks by 🔴/🟠 status **and** distance from head; give **tail-end + high-stress chaks priority** in the roster to offset conveyance/seepage losses (lower effective `Ea` at the tail). Report **volume `V`, turn time `V/Q`, and a priority index** per outlet.

---

## 8. Stress-Index Cross-Checks — the publishing gate

Before an advisory is published for a command area, **four independent stress signals must tell one coherent story** (the "credible command-area planning" rule):

| Index | Formula / basis | Says "stressed" when |
|---|---|---|
| **VCI** (Veg. Condition) | `VCI = (NDVI − NDVI_min)/(NDVI_max − NDVI_min)·100` | VCI low (<35) — canopy below normal |
| **TVDI** (Temp-Veg Dryness) | `TVDI = (Ts − Ts_min)/(Ts_max(NDVI) − Ts_min)` (Ts–NDVI "triangle") | TVDI → 1 (dry edge) |
| **CWSI** (Crop Water Stress) | `CWSI = (dT − dT_LL)/(dT_UL − dT_LL)`, `dT=Tc−Ta` | CWSI → 1 (canopy hot vs well-watered limit) |
| **SM anomaly** | z-score of SMAP/ASCAT/EOS-04 θ vs climatology | strongly negative |

**Gate:** modeled `Ks<1`, satellite **ETa depression**, **CWSI/TVDI high**, **VCI low**, and **negative SM anomaly** should **co-occur**. If they **disagree** (e.g. low Ks but green VCI and wet SM ⇒ likely Kc/percolation error, or recent un-modeled rain), **flag for review — do not issue an irrigation order.** Coherence across all streams is what makes a command-area plan defensible to a canal authority.

---

## 9. India Context — Schemes, Portals, Data

**Programs the advisory plugs into:**
- **PMKSY** (Pradhan Mantri Krishi Sinchayee Yojana — "Per Drop More Crop"; on-farm efficiency, micro-irrigation `Ea`↑)
- **PMFBY** (crop insurance — ETa/stress maps support loss assessment)
- **Digital Agriculture Mission / AgriStack**, **NMSA** (National Mission for Sustainable Agriculture), **FASAL** (crop forecasting), **NICES** (ISRO ecosystem/veg services)

**Data portals (ISRO/IMD):**
- **MOSDAC** (SAC) — INSAT-3D insolation/LST, gridded **ET₀**, atmospheric products
- **Bhuvan / Bhoonidhi (NRSC)** — Resourcesat NDVI, **EOS-04 500 m soil moisture**, **NISAR 100 m**, land use, satellite archive
- **IMD gridded rainfall (0.25°)/temperature** — `P` and benchmark `T`
- **India-WRIS** — **canal command-area boundaries**, reservoirs, irrigation infrastructure (the spatial frame for §7.5)

---

## 10. Recommended Pipeline

```
                    ┌──────────────────────────────────────────────────────────────┐
 WEATHER GRID  ───► │ FAO-56 Penman-Monteith                                        │
 (AgERA5 / IMD-GFS  │   Δ, es, ea, γ, Rn=Rns−Rnl  ──►  ET0  (~12.5 km daily, +8d fc)│
  + INSAT-3D Rs)    └───────────────┬──────────────────────────────────────────────┘
                                    │
 NDVI (S2/Landsat/  ─► Kc=1.457·NDVI−0.1725  /  SIMS fc→Kd→Kcb
  Resourcesat) ──────────────────►  │
                                    ▼
                          ETc = (Kcb+Ke)·ET0   ◄── climate-adjust Kc_mid; +rice percolation
                                    │
 THERMAL LST  ─► Energy balance ─► ETa (SSEBop/METRIC/PT-JPL … OpenET-style ensemble)
 (Landsat/EOS-04/         │              │  (independent observation)
  INSAT morning rise)     │              │
                          ▼              ▼
 SOIL TEXTURE ─► θFC,θWP ─► TAW=1000(θFC−θWP)Zr ;  RAW=p·TAW
 (SoilGrids/NBSS,                        │
  Saxton-Rawls)                          ▼
 SOIL MOISTURE ─► SWI exp-filter ─► θ_RZ ──► INITIALIZE / ASSIMILATE (nudge/EnKF) ─┐
 (SMAP/ASCAT/EOS-04/NISAR)                                                          │
                                    ▼                                              │
                ┌──────────────────────────────────────────────────────┐          │
                │ ROOT-ZONE WATER BALANCE (FAO-56 Ch.8, daily)          │ ◄────────┘
                │ Dr,i = Dr,i−1 −(P−RO)−I−CR +ETc,adj +DP               │
                │ Ks=(TAW−Dr)/((1−p)TAW) ;  ETc,adj=Ks·Kc·ET0           │
                └───────────────┬──────────────────────────────────────┘
                                ▼
        8-DAY DEFICIT  Deficit_8d = ΣETc,adj − (ΣPeff+ΔS) = Dr  ;  dnet=Dr ; dgross=dnet/Ea
                                │
            STRESS CROSS-CHECK (VCI·TVDI·CWSI·SM anomaly coherent?)  ──► gate (§8)
                                ▼
        STAGE-AWARE ADVISORY  (5-class 🟢🔵🟡🟠🔴 ; when=Dr→RAW ; how much=dgross ; rice AWD)
                                ▼
        COMMAND AGGREGATION  pixel→field→chak→distributary ; V=Σdgross·A ; turn=V/Q ;
                             warabandi 8-day roster ; tail-end/equity priority
```

---

## 11. Worked Numerical Example

**Setting:** Wheat, **mid-season** (flowering), **sandy-loam** soil, a canal chak of **20 ha**, surface irrigation.

**Step 1 — Available water.**
θFC=0.21, θWP=0.09, Zr=1.0 m ⇒
`TAW = 1000·(0.21−0.09)·1.0 = 120 mm`.
Default `p=0.5` (mid-season; tightened from 0.55 because flowering is sensitive) ⇒
`RAW = 0.5·120 = 60 mm`.

**Step 2 — Crop demand.**
`ET0 = 6 mm/d` (hot, dry NW-India spring), wheat `Kc_mid=1.15` ⇒
`ETc = 1.15·6 ≈ 6.9 mm/d` (so long as `Ks=1`).

**Step 3 — Depletion accumulation (assume start at field capacity, no rain in window):**

| Day | ETc,adj (mm) | Cumulative Dr (mm) | Dr vs RAW/TAW | Ks | Status |
|---|---|---|---|---|---|
| 0 | — | 0 | — | 1.00 | 🟢 |
| 1 | 6.9 | 6.9 | < ½RAW | 1.00 | 🟢 |
| 4 | 6.9 | 27.6 | ≈ ½RAW (30) | 1.00 | 🟢→🔵 |
| 5 | 6.9 | 34.5 | > ½RAW | 1.00 | 🔵 Watch |
| **8** | 6.9 | **55.2** | just < RAW (60) | 1.00 | 🔵→🟡 (trigger imminent) |
| ~9 | — | **60** | **= RAW** | **1.00→<1** | 🟡 **Irrigate-soon** (TRIGGER) |

So the **8-day deficit ≈ 55 mm**, and the field reaches the **RAW trigger (`Dr=60`) on ~day 9** — i.e. **schedule the canal turn at the end of this 8-day window**, which is exactly when the next warabandi rotation comes around.

**Step 4 — Stress if irrigation is missed.** Suppose delivery slips and depletion reaches **`Dr=90 mm`** (past RAW=60, below TAW=120):
```
Ks = (TAW − Dr)/((1−p)·TAW) = (120 − 90)/((1−0.5)·120) = 30/60 = 0.50
```
`Ks=0.50` ⇒ 🟠 **Irrigate-now** (actual ET halved, ~50% yield-stress) — and we'd expect satellite **CWSI high, VCI dropping, SMAP θ-anomaly negative, ETa ≈ 0.5·ETc** to *all agree* before publishing this alert (§8).

**Step 5 — Irrigation depths (refill from `Dr=60` at trigger).**
`dnet = Dr = 60 mm`. Surface efficiency `Ea=0.60` ⇒
```
dgross = dnet / Ea = 60 / 0.60 = 100 mm
```
*(If we waited to Dr=90: dnet=90, dgross=150 mm — illustrating the cost of late irrigation.)*

**Step 6 — Command-area volume & turn time (chak = 20 ha):**
```
V = dgross · A = 0.100 m · 200,000 m² = 20,000 m³
```
At outlet discharge `Q = 30 L/s = 0.03 m³/s`:
```
turn time t = V / Q = 20,000 / 0.03 ≈ 666,700 s ≈ 185 hours ≈ 7.7 days of continuous flow
```
⇒ This chak needs essentially its **full ~8-day warabandi slot** at this outlet capacity. Cross-check vs the **1 h/acre/week** rule: 20 ha ≈ 49.4 acre ⇒ ~49 h/week nominal — the **185 h** here reflects the *high* 100 mm gross dose (peak demand, low surface efficiency), flagging this chak for **priority and/or efficiency intervention (PMKSY micro-irrigation: drip Ea=0.90 ⇒ dgross=67 mm ⇒ V=13,300 m³ ⇒ ~123 h, a 33% saving).**

---

## References

1. **Allen, R.G., Pereira, L.S., Raes, D., Smith, M. (1998).** *Crop Evapotranspiration — Guidelines for computing crop water requirements.* **FAO Irrigation & Drainage Paper 56**, FAO, Rome. (ET₀ PM eq., Kc tables, Ch.8 water balance, p, Ks, Peff.)
2. **ASCE-EWRI (2005).** *The ASCE Standardized Reference Evapotranspiration Equation.* (Cn/Cd standardized form.)
3. **Pereira, L.S., Paredes, P., Jovanovic, N. et al. (2021+).** *Updating FAO-56 Kc with GDD / remote sensing (FAO-56 revisited).* Agric. Water Manag.
4. **Senay, G.B. et al. (2013; 2018).** *Operational Simplified Surface Energy Balance (SSEBop)* — incl. *Satellite Psychrometric Formulation (ETf=(Th−Ts)/dT).* ASCE/JAWRA. https://elibrary.asabe.org/abstract.asp?AID=48975
5. **Bastiaanssen, W.G.M. (1998) SEBAL; Allen, R.G. (2007) METRIC.** J. Hydrology / J. Irrig. Drain. Eng.
6. **Melton, F.S. et al. (2023).** *Assessing the accuracy of OpenET satellite-based evapotranspiration data.* **Nature Water** 1, 233–250. https://www.nature.com/articles/s44221-023-00181-7 (6-model ensemble, r²≈0.90, MAE 0.74–1.07 mm/d.)
7. **Wagner, W., Lemoine, G., Rott, H. (1999).** *A method for estimating soil moisture from ERS scatterometer and soil data* (exponential filter / **Soil Water Index**). Remote Sens. Environ. 70, 191–207. (SWI characteristic time T.)
8. **Saxton, K.E., Rawls, W.J. (2006).** *Soil water characteristic estimates by texture and organic matter (pedotransfer).* SSSAJ 70, 1569–1578.
9. **Kamble, B., Kilic, A., Hubbard, K. (2013).** *Estimating crop coefficients using remote sensing-based vegetation index (Kc=1.457·NDVI−0.1725, r²≈0.90).* Remote Sensing 5, 1588–1602.
10. **Pereira/Allen — SIMS (fc→Kd→Kcb) & OpenET-SIMS (Melton/Pereira).**
11. **Bhattacharya, B.K. et al. / SAC-ISRO — Near-real-time gridded reference ET₀ over India (FAO-56 PM fused with INSAT/Kalpana insolation + GFS).** *Earth Science Informatics* (Springer) 2023. https://link.springer.com/article/10.1007/s12145-023-01197-z
12. **SMAP L4 RZSM (Reichle et al.); ASCAT SWI (Copernicus Global Land); EOS-04 / NISAR soil moisture — ISRO Bhoonidhi/MOSDAC.**
13. **USDA-SCS** — Effective rainfall & **Curve Number** runoff method (TR-55 / NEH-4).
14. **Sources confirmed via web search (June 2026):** SSEBop psychrometric formulation (USGS/ASABE); OpenET *Nature Water* 2023 accuracy; SWI exponential filter (Wagner 1999, Copernicus); India 12.5 km ET₀ grid (Springer ESI 2023).
```
