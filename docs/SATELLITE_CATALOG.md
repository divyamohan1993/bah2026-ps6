# AgriStress Satellite & Sensor Data Catalog

**ISRO BAH 2026 — Problem Statement 6: Crop-type Classification + Moisture-Stress Detection + Irrigation Advisory**

> Authoritative multi-satellite, multi-sensor data catalog for the AgriStress fusion system. This document enumerates every Earth-observation asset the system can ingest — 40+ satellites/sensors plus ancillary climate, land-cover, and embedding grids — organised into 9 categories, with verified cloud asset IDs, access portals, cost, and an explicit statement of the role each plays and the failure mode it covers.

**Maintainer:** divyamohan1993@gmail.com
**Last verified:** 2026-06-15
**Scope:** India (kharif + rabi cropping seasons), with the architecture generalisable to any agro-climatic zone.

---

## 0. How to read this catalog

- **Native res** — finest ground sampling distance of the primary agricultural bands (optical VNIR/SWIR, SAR, or microwave footprint).
- **Revisit** — effective repeat at the equator for a single platform; constellations (e.g. S2 A+B+C) improve this proportionally.
- **Cloud asset ID** — the canonical identifier used to load the collection. `COPERNICUS/...`, `LANDSAT/...`, `MODIS/...`, `NASA/...`, `GOOGLE/...`, `JAXA/...`, `EO1/...`, `USGS/...`, `ECMWF/...`, `ESA/...`, `UCSB-CHG/...`, `TRMM/...` and `NOAA/...` IDs are **Google Earth Engine (GEE)** asset paths. `ECO_*`, `VNP*`/`MOD*`/`MYD*` (when LP DAAC), `EMIT_*`, `NISAR_*`, `PRS_*`, `ENMAP_*` and commercial IDs are **external-ingestion** product short-names (Microsoft Planetary Computer STAC, NASA Earthdata/LP DAAC, ASF DAAC, ISRO Bhoonidhi, ISRO/MOSDAC, JAXA G-Portal, DLR EOWEB). See §13 (GEE-native vs external-ingestion).
- **Cost** — Free = open/public; Free* = open but requires registration/EULA; Tasking/Commercial = paid (price varies by tasking vs archive, AOI, and reseller).
- A **HEADLINE** tag marks NISAR — the flagship NASA-ISRO dual-frequency SAR asset central to PS6.

---

## 1. OPTICAL — Multispectral (VNIR/SWIR) wide-swath workhorses

| # | Name | Operator / Country | Sensor type | Bands / Pol | Native res | Revisit | Status | Cloud asset ID | Portal | Cost | Role in crop / moisture / irrigation system | Gap it fills |
|---|------|--------------------|-------------|-------------|-----------|---------|--------|----------------|--------|------|----------------------------------------------|--------------|
| 1 | Sentinel-2A MSI | ESA/Copernicus (EU) | Optical multispectral | 13 bands (4×10 m, 6×20 m, 3×60 m); B2-B12; 3 red-edge | 10 m | 10 d (5 d with S2B) | Operational | `COPERNICUS/S2_SR_HARMONIZED` | GEE / Copernicus DataSpace | Free | Primary NDVI/EVI/NDRE/NDWI/NDMI engine for crop-type & vigour; red-edge drives N & chlorophyll stress | High-res optical backbone of the whole system |
| 2 | Sentinel-2B MSI | ESA/Copernicus (EU) | Optical multispectral | Same as S2A | 10 m | 10 d (5 d combined) | Operational | `COPERNICUS/S2_SR_HARMONIZED` | GEE / Copernicus | Free | Doubles S2 cadence; cloud-gap reduction in monsoon | Halves optical revisit |
| 3 | Sentinel-2C MSI | ESA/Copernicus (EU) | Optical multispectral | Same as S2A | 10 m | adds to 5 d (→ ~3-4 d 3-sat) | Operational (commissioned 2025, replaced S2A ops slot) | `COPERNICUS/S2_SR_HARMONIZED` | GEE / Copernicus | Free | Extends MSI continuity; tightens revisit during clouds | Constellation continuity & denser time-series |
| 4 | Landsat 9 OLI-2 / TIRS-2 | USGS-NASA (USA) | Optical + thermal | 11 bands; VNIR/SWIR 30 m, pan 15 m, TIR 100 m | 30 m (VNIR/SWIR) | 16 d (8 d with L8) | Operational | `LANDSAT/LC09/C02/T1_L2` | GEE / USGS EE | Free | Surface-reflectance crop indices + **surface temperature (ST)** band for ET/stress; cross-calibrated with S2 | Adds thermal + a second optical source for cloud gap-fill |
| 5 | Landsat 8 OLI / TIRS | USGS-NASA (USA) | Optical + thermal | 11 bands; 30 m VNIR/SWIR, 100 m TIR | 30 m | 16 d (8 d with L9) | Operational | `LANDSAT/LC08/C02/T1_L2` | GEE / USGS EE | Free | Long, calibrated ST + reflectance archive; harmonised with S2 (HLS) | Thermal stress + 12-yr baseline for anomaly detection |
| 6 | Landsat 7 ETM+ | USGS-NASA (USA) | Optical + thermal | 8 bands, 30 m, 60 m TIR | 30 m | 16 d | Decommissioned 2024 (archive only; SLC-off post-2003) | `LANDSAT/LE07/C02/T1_L2` | GEE / USGS EE | Free | Historical baseline for multi-decadal cropland/ET trends | Extends time-series back to 1999 |
| 7 | Landsat 5 TM | USGS-NASA (USA) | Optical + thermal | 7 bands, 30 m, 120 m TIR | 30 m | 16 d | Retired (archive only) | `LANDSAT/LT05/C02/T1_L2` | GEE / USGS EE | Free | Deep historical baseline (1984-2012) | Longest dense optical record for climatology |
| 8 | MODIS (Terra) | NASA (USA) | Optical/thermal wide-swath | 36 bands; 250 m / 500 m / 1 km | 250 m | Daily | Operational (ageing) | `MODIS/061/MOD13Q1` (VI), `MODIS/061/MOD09GA` (SR) | GEE | Free | Daily 250 m NDVI/EVI for phenology, drought (VCI/TCI), large-area screening | Daily revisit nobody else gives at moderate res |
| 9 | MODIS (Aqua) | NASA (USA) | Optical/thermal wide-swath | 36 bands | 250 m | Daily | Operational (ageing) | `MODIS/061/MYD13Q1`, `MODIS/061/MYD09GA` | GEE | Free | Afternoon overpass complements Terra; 2×/day VI/LST sampling | Sub-daily land sampling |
| 10 | VIIRS (Suomi-NPP / NOAA-20/21) | NOAA-NASA (USA) | Optical/thermal wide-swath | 22 bands; 375 m / 750 m | 375 m | Daily | Operational | `NOAA/VIIRS/001/VNP09GA` (SR), `NASA/VIIRS/002/VNP13A1` (VI) | GEE | Free | MODIS successor: daily VI, active-fire, LST continuity into 2030s | Operational continuity after MODIS EOL |
| 11 | Sentinel-3 OLCI | ESA/Copernicus (EU) | Optical (ocean/land colour) | 21 bands, 300 m | 300 m | ~1-2 d (A+B) | Operational | `COPERNICUS/S3/OLCI` | GEE | Free | Wide-swath chlorophyll/FAPAR; fluorescence-adjacent bands for vegetation status | 300 m daily land-colour bridge between MODIS and S2 |
| 12 | PROBA-V | ESA / VITO (EU) | Optical wide-swath | 4 bands (B,R,NIR,SWIR); 100-333 m | 100 m | Daily (1 d global) | Archive (ops ended 2021; 100 m archive live) | `VITO/PROBAV/C1/S1_TOC_100M` | GEE / VITO PV-MEP | Free | SPOT-VGT continuity; 100 m near-daily VI for the 2013-2020 gap-fill | Bridges MODIS-class and decametric optical historically |
| 13 | Resourcesat-2 / 2A — LISS-III | ISRO (India) | Optical multispectral | 4 bands (G,R,NIR,SWIR), 23.5 m | 23.5 m | 24 d (5 d combined w/ AWiFS) | Operational | `ISRO/RESOURCESAT/LISS3` (Bhoonidhi product) | Bhoonidhi (NRSC) | Free* | National-context Indian optical; calibrated for Indian crop signatures, DES/GCES schemes | Sovereign Indian optical aligned with FASAL/CCE protocols |
| 14 | Resourcesat-2 / 2A — LISS-IV | ISRO (India) | Optical multispectral | 3 bands (G,R,NIR), 5.8 m | 5.8 m | 5 d (mono mode) | Operational | Bhoonidhi product (`RS2_LISS4`) | Bhoonidhi (NRSC) | Free* | 5.8 m field-level crop discrimination, plot boundaries | Sub-10 m Indian optical for smallholder fields |
| 15 | Resourcesat-2 / 2A — AWiFS | ISRO (India) | Optical wide-swath | 4 bands, 56 m, 740 km swath | 56 m | 5 d | Operational | Bhoonidhi product (`RS2_AWIFS`) | Bhoonidhi (NRSC) | Free* | State/district-scale rapid coverage; kharif/rabi acreage | Wide-swath sovereign coverage at 5-day repeat |
| 16 | Cartosat-2 series | ISRO (India) | Optical pan/MS | Pan ≤1 m, MS ~2 m | <1 m (pan) | On-demand | Operational | Bhoonidhi product | Bhoonidhi (NRSC) | Free*/Tasking | Sub-meter validation of plot boundaries, encroachment, canal mapping | Indian sub-meter context & validation |
| 17 | Cartosat-3 | ISRO (India) | Optical pan/MS | Pan 0.25 m, MS 1.13 m, SWIR | 0.25 m (pan) | On-demand | Operational | Bhoonidhi product | Bhoonidhi (NRSC) | Free*/Tasking | Highest-res Indian optical; ground-truth & irrigation-infrastructure audit | Sovereign 25 cm validation imagery |
| 18 | GaoFen-1 | CNSA / CRESDA (China) | Optical multispectral | Pan 2 m, MS 8 m, 16 m WFV | 8 m / 16 m | 4 d (16 m WFV) | Operational | External (CRESDA) | CRESDA / partner archives | Free*/Commercial | Independent regional cross-check; trans-boundary basin coverage | Extra optical revisit + geopolitical independence |
| 19 | GaoFen-6 | CNSA / CRESDA (China) | Optical multispectral | 2 m pan / 8 m MS + 16 m WFV (red-edge) | 8 m | 4 d | Operational | External (CRESDA) | CRESDA | Free*/Commercial | Red-edge WFV tuned for agriculture; wide-area crop mapping | Red-edge wide-swath alternative to S2 |
| 20 | Himawari-8 / 9 AHI | JMA / JAXA (Japan) | Geostationary optical/IR | 16 bands, 0.5-2 km | 0.5 km (VIS) | **10 min** (full disk) | Operational | External (JAXA — **not on GEE**) | JAXA Himawari Monitor / P-Tree | Free* | Diurnal cloud-clearing scheduler; sub-hourly LST/cloud to pick clear S2/S1 windows | Geostationary temporal density over Asia (cloud timing) |
| 21 | GOES-16 (East) ABI | NOAA (USA) | Geostationary optical/IR | 16 bands, 0.5-2 km | 0.5 km | 5-15 min | Operational | `NOAA/GOES/16/MCMIPF` | GEE | Free | Cloud/aerosol/fire context (Americas); geostationary fusion template | Geostationary nowcasting reference |
| 22 | GOES-18 (West) ABI | NOAA (USA) | Geostationary optical/IR | 16 bands, 0.5-2 km | 0.5 km | 5-15 min | Operational | `NOAA/GOES/18/FDCF` (fire), `NOAA/GOES/18/MCMIPF` | GEE | Free | Active-fire detection product; smoke/haze masking | Sub-hourly fire & haze flags |
| 23 | INSAT-3D / 3DR / 3DS | ISRO (India) | Geostationary imager + sounder | 6 imager ch (VIS, SWIR, MIR, WV, 2×TIR) | 1 km (VIS) | **15-30 min** | Operational (3DS launched 2025) | External (MOSDAC — **not on GEE**) | MOSDAC (SAC/ISRO) | Free* | Sovereign geostationary: diurnal LST, INSAT rainfall, fog/cloud over India for acquisition planning | Indian geostationary cloud + thermal + rainfall, 30-min |

---

## 2. COMMERCIAL HIGH-RESOLUTION — Tasking & dense daily

| # | Name | Operator / Country | Sensor type | Bands / Pol | Native res | Revisit | Status | Cloud asset ID | Portal | Cost | Role in crop / moisture / irrigation system | Gap it fills |
|---|------|--------------------|-------------|-------------|-----------|---------|--------|----------------|--------|------|----------------------------------------------|--------------|
| 24 | PlanetScope (Doves / SuperDoves) | Planet Labs (USA) | Optical multispectral | 4-8 bands (+red-edge on SuperDove) | 3 m | **Daily** | Operational | `PLANET/SCOPE` (external STAC) | Planet API / NICFI | Commercial (NICFI tropics free) | Daily 3 m fills every cloud gap; field-scale stress between S2 passes | Daily decimetre-to-3 m gap-killer |
| 25 | RapidEye | Planet (archive) | Optical multispectral | 5 bands (incl. red-edge) | 5 m | ~5.5 d | Archive (2009-2020) | External (Planet) | Planet API | Commercial | First operational red-edge constellation; historical 5 m | Pre-2020 red-edge archive |
| 26 | SkySat | Planet Labs (USA) | Optical pan/MS + video | Pan 0.5 m, 4 MS bands | 0.5 m | Sub-daily (tasking) | Operational | External (Planet) | Planet API | Tasking | On-demand sub-meter for dispute/insurance/validation events | Tasked sub-meter snapshots |
| 27 | SPOT-6 / 7 | Airbus DS (EU) | Optical pan/MS | Pan 1.5 m, 4 MS bands | 1.5 m | Daily (tasking) | Operational | External (Airbus) | Airbus OneAtlas | Commercial | Regional 1.5 m mapping; canal/field infrastructure | 1.5 m wide-area commercial tier |
| 28 | Pléiades 1A/1B | Airbus DS (EU) | Optical pan/MS | Pan 0.5 m, 4 MS bands | 0.5 m | Daily (tasking) | Operational | External (Airbus) | Airbus OneAtlas | Tasking | 50 cm validation & high-value plot audit | Reliable 50 cm tasking |
| 29 | Pléiades Neo | Airbus DS (EU) | Optical pan/MS | Pan 0.3 m, 6 MS bands (+red-edge, deep-blue) | 0.3 m | Sub-daily (tasking) | Operational | External (Airbus) | Airbus OneAtlas | Tasking | 30 cm + 6-band agronomic detail; tree/row crops | Finest commercial multispectral |
| 30 | WorldView-2 | Maxar (USA) | Optical multispectral | 8 bands (incl. red-edge, 2 NIR) | 0.46 m | ~1.1 d (tasking) | Operational | External (Maxar) | Maxar / SecureWatch | Tasking | 8-band sub-meter for detailed crop & weed mapping | 8-band sub-meter spectral richness |
| 31 | WorldView-3 (+ SWIR) | Maxar (USA) | Optical MS + SWIR | 8 VNIR + **8 SWIR** + 12 CAVIS | 0.31 m (pan), 3.7 m SWIR | ~1 d (tasking) | Operational | External (Maxar) | Maxar / SecureWatch | Tasking | Only commercial sat with 8-band SWIR — soil/residue/moisture surrogates at 3.7 m | Commercial SWIR for soil & residue |
| 32 | GeoEye-1 | Maxar (USA) | Optical pan/MS | Pan 0.41 m, 4 MS bands | 0.41 m | ~3 d (tasking) | Operational | External (Maxar) | Maxar / SecureWatch | Tasking | Additional sub-meter tasking capacity / stereo | Sub-meter tasking redundancy |

---

## 3. HYPERSPECTRAL — Hundreds of contiguous bands (biochemistry & soil)

| # | Name | Operator / Country | Sensor type | Bands / Pol | Native res | Revisit | Status | Cloud asset ID | Portal | Cost | Role in crop / moisture / irrigation system | Gap it fills |
|---|------|--------------------|-------------|-------------|-----------|---------|--------|----------------|--------|------|----------------------------------------------|--------------|
| 33 | PRISMA | ASI (Italy) | Hyperspectral + pan | **~240 bands** (VNIR+SWIR, 400-2500 nm) | 30 m (HS), 5 m pan | ~7 d (tasking) | Operational | External (`PRS_L2D`); also via Bhoonidhi (India AOIs) | ASI / Bhoonidhi | Free* | Full spectral signatures: species ID, leaf chemistry, soil clay/moisture absorption features | Spaceborne hyperspectral discrimination |
| 34 | EnMAP | DLR / GFZ (Germany) | Hyperspectral | **~246 bands** (420-2450 nm) | 30 m | ~4 d (tasking), 27 d nadir | Operational | External (`ENMAP_HSI_L2A`) | DLR EOWEB GeoPortal | Free* | High-SNR hyperspectral; crop trait retrieval, soil organic carbon, water-stress red-edge inflection | High-quality HS for trait & soil mapping |
| 35 | EMIT | NASA-JPL / ISS (USA) | Imaging spectrometer | **285 bands** (380-2500 nm) | 60 m | ~3 d (ISS, variable) | Operational | `EMIT_L2A_RFL` | LP DAAC / **Planetary Computer (STAC)** | Free* | ISS-borne VSWIR for mineralogy, soil moisture absorption, canopy water content | Free frequent hyperspectral over India |
| 36 | DESIS | DLR / Teledyne — ISS (Germany/USA) | Hyperspectral (VNIR) | 235 bands (400-1000 nm) | 30 m | ISS-variable | Archive (ops ended 2023) | External (Teledyne / DLR) | DLR EOWEB / Teledyne | Free*/Commercial | VNIR hyperspectral archive for red-edge & pigment retrieval | VNIR HS archive for chlorophyll/anthocyanin |
| 37 | Hyperion (EO-1) | NASA (USA) | Hyperspectral | 220 calibrated bands (400-2500 nm) | 30 m | 16 d (tasking) | Retired (archive 2001-2017) | `EO1/HYPERION` | GEE | Free | First spaceborne HS archive — algorithm prototyping & historical signatures | GEE-native HS for method development |

---

## 4. SAR — All-weather, day-night radar (cloud-immune)

| # | Name | Operator / Country | Sensor type | Bands / Pol | Native res | Revisit | Status | Cloud asset ID | Portal | Cost | Role in crop / moisture / irrigation system | Gap it fills |
|---|------|--------------------|-------------|-------------|-----------|---------|--------|----------------|--------|------|----------------------------------------------|--------------|
| 38 | Sentinel-1A | ESA/Copernicus (EU) | SAR | **C-band**, VV+VH (IW) | 10 m (GRD) | 12 d (6 d w/ S1C) | Operational | `COPERNICUS/S1_GRD` | GEE / Copernicus | Free | All-weather crop structure, backscatter phenology, flood/standing-water, soil-moisture proxy | Cloud-proof backbone — the monsoon workhorse |
| 39 | Sentinel-1C | ESA/Copernicus (EU) | SAR | C-band, VV+VH | 10 m | restores 6 d w/ S1A | Operational (launched Dec 2024, replaced S1B; nominal 2025) | `COPERNICUS/S1_GRD` | GEE / Copernicus | Free | Re-establishes 6-day C-band after S1B loss; denser InSAR/coherence | Restores dense SAR cadence |
| 40 | **NISAR** | **NASA + ISRO (USA/India)** | **Dual-frequency SAR** | **L-band + S-band**, quad/dual-pol; **242 km swath** | 3-10 m | **12 d** (exact-repeat) | **Operational — launched 30 Jul 2025; L-band products on ASF DAAC (100k+ files released)** | External: `NISAR_L1/L2` (ASF DAAC) + Bhoonidhi (S-band, India) | **ASF DAAC** + Bhoonidhi | Free* | **HEADLINE.** L-band penetrates canopy → biomass, soil moisture, deep crop structure; S-band complements; systematic 12-day all-India coverage for stress & subsidence | First systematic dual-freq SAR — fills the L-band soil/biomass gap entirely |
| 41 | EOS-04 / RISAT-1A | ISRO (India) | SAR | **C-band**, multi-pol | ~3-25 m (modes) | ~12 d | Operational (2022) | External (Bhoonidhi) | Bhoonidhi (NRSC) | Free* | Sovereign C-band SAR — kharif rice/standing-water, soil moisture over India | Indian C-band SAR independence |
| 42 | RISAT-2B / 2BR1 | ISRO (India) | SAR | **X-band**, multi-pol | ≤1 m (HRS modes) | On-demand | Operational | External (Bhoonidhi) | Bhoonidhi (NRSC) | Free*/Tasking | High-res X-band for field structure, infrastructure, rapid tasking | Indian high-res X-band SAR |
| 43 | ALOS-2 PALSAR-2 | JAXA (Japan) | SAR | **L-band**, quad-pol | 3-10 m; 25 m mosaic | 14 d | Operational | `JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH` (annual L-band mosaic) | GEE (mosaic) / JAXA (scenes) | Free (mosaic) | Pre-NISAR L-band reference & continuity; annual L-band backscatter baseline | Established L-band archive/mosaic on GEE |
| 44 | RADARSAT-2 | MDA / CSA (Canada) | SAR | C-band, quad-pol | 1-100 m | 24 d | Operational | External (MDA) | MDA / CSA | Commercial | High-quality polarimetric C-band; agronomic decomposition | Quad-pol C-band tasking |
| 45 | RCM (RADARSAT Constellation) | CSA (Canada) | SAR (×3) | C-band, compact-pol | 3-100 m | **4 d** (3-sat) | Operational | External (CSA) | CSA EODMS | Free*/Commercial | 4-day C-band coverage; compact-pol soil-moisture & change | Dense C-band revisit alternative |
| 46 | TerraSAR-X / TanDEM-X | DLR / Airbus (Germany) | SAR | **X-band**, multi-pol | ≤1 m (SL) | 11 d | Operational | External (Airbus / DLR) | Airbus / DLR | Commercial | Ultra-high-res X-band; bistatic DEM (source of CopDEM); precise field structure | Sub-meter X-band + global DEM source |
| 47 | COSMO-SkyMed (1st+2nd gen) | ASI / Italian MoD | SAR (×4) | X-band, multi-pol | ≤1 m | Sub-daily (4-sat) | Operational | External (ASI / e-GEOS) | ASI / e-GEOS | Commercial | Rapid X-band tasking; flood-irrigation mapping | High-cadence X-band emergency tasking |
| 48 | SAOCOM 1A/1B | CONAE (Argentina) | SAR | **L-band**, quad-pol | 10 m | 16 d (8 d, 2-sat) | Operational | External (CONAE) | CONAE | Free*/Commercial | L-band soil-moisture products (agronomic heritage); complements NISAR/PALSAR | Operational L-band soil-moisture service |
| 49 | GaoFen-3 | CNSA (China) | SAR | C-band, multi-pol | 1-500 m | ~3 d (with successors) | Operational | External (CRESDA) | CRESDA | Free*/Commercial | Independent C-band; trans-boundary basins | Extra C-band, geopolitical independence |
| 50 | ICEYE (constellation) | ICEYE (Finland) | SAR (NewSpace) | X-band | ≤0.25 m (spot) | Sub-daily / hours | Operational | External (ICEYE) | ICEYE API | Commercial | Persistent X-band tasking; rapid flood & damage; very high revisit | On-demand hourly SAR |
| 51 | Capella Space | Capella (USA) | SAR (NewSpace) | X-band | ≤0.5 m | Sub-daily | Operational | External (Capella) | Capella Console | Commercial | High-res X-band tasking redundancy | Commercial X-band surge capacity |

---

## 5. DEDICATED SOIL MOISTURE — Microwave radiometers/scatterometers

| # | Name | Operator / Country | Sensor type | Bands / Pol | Native res | Revisit | Status | Cloud asset ID | Portal | Cost | Role in crop / moisture / irrigation system | Gap it fills |
|---|------|--------------------|-------------|-------------|-----------|---------|--------|----------------|--------|------|----------------------------------------------|--------------|
| 52 | SMAP L4 (surface + root-zone) | NASA (USA) | L-band radiometer + model assimilation | L-band (1.4 GHz) | 9 km | 3-hourly (daily mean) | Operational | `NASA/SMAP/SPL4SMGP/008` | GEE | Free | **Root-zone (0-100 cm)** + surface (0-5 cm) soil moisture — the irrigation-decision variable; assimilated/gap-free | The only spaceborne root-zone soil-moisture source |
| 53 | SMAP L3 (enhanced) | NASA (USA) | L-band radiometer | L-band | 9 km | 2-3 d | Operational | `NASA/SMAP/SPL3SMP_E/006` | GEE | Free | Direct retrieved surface soil moisture (less modelling than L4) | Observation-driven surface SM truth |
| — | ~~SMAP-USDA 10 km~~ | NASA-USDA | L-band (derived) | L-band | 10 km | 3 d | **DEPRECATED — do not use** | ~~`NASA_USDA/HSL/SMAP10KM_soil_moisture`~~ | (GEE, deprecated) | — | Superseded by `SPL4SMGP/008` + `SPL3SMP_E/006`. Listed only to prevent accidental reuse. | (none — historical) |
| 54 | SMOS | ESA / CNES / CDTI (EU) | L-band interferometric radiometer (MIRAS) | L-band | ~35-40 km | 2-3 d | Operational | External (CATDS / ESA) | CATDS / ESA | Free* | Independent L-band SM + soil freeze; cross-validates SMAP; longest L-band record (2010-) | Second independent L-band SM (de-risks single mission) |
| 55 | ASCAT (MetOp-B / C) | EUMETSAT (EU) | C-band scatterometer | C-band | 25 km (12.5 km SSM) | Daily | Operational | External (EUMETSAT / Copernicus SWI) | Copernicus Global Land / EUMETSAT | Free* | Daily Soil Water Index (SWI) profile; long climate record; scatterometer cross-check | Daily C-band SM with deep climatology |
| 56 | AMSR2 (GCOM-W) | JAXA (Japan) | Multi-freq microwave radiometer | C/X/K/Ka-band | ~10-50 km | 1-2 d | Operational | External (JAXA / LPRM) | JAXA G-Portal / GES DISC (LPRM) | Free* | LPRM soil moisture + land surface temp + vegetation optical depth (VOD) for biomass/water | Multi-frequency SM + VOD biomass proxy |
| 57 | Sentinel-1 SSM 1 km | ESA / Copernicus Global Land (EU) | SAR-derived surface SM | C-band-derived | **1 km** | ~1.5-4 d | Operational | External (Copernicus Global Land Service) | land.copernicus.eu | Free | **1 km** surface soil moisture — bridges 9 km SMAP and field scale | Decametric-to-km SM downscaling reference |

---

## 6. PRECIPITATION — Rainfall forcing for water-balance & irrigation need

| # | Name | Operator / Country | Sensor type | Bands / Pol | Native res | Revisit | Status | Cloud asset ID | Portal | Cost | Role in crop / moisture / irrigation system | Gap it fills |
|---|------|--------------------|-------------|-------------|-----------|---------|--------|----------------|--------|------|----------------------------------------------|--------------|
| 58 | GPM IMERG V07 | NASA / JAXA (USA/Japan) | Multi-sat passive-MW + IR merge | Precipitation | ~10 km (0.1°) | 30 min | Operational | `NASA/GPM_L3/IMERG_V07` | GEE | Free | Half-hourly global rainfall — antecedent-precipitation index, water balance, dry-spell detection | Best-resolution global near-real-time rainfall |
| 59 | TRMM 3B42 | NASA / JAXA | Passive-MW + IR | Precipitation | ~25 km (0.25°) | 3 h | Retired (archive 1998-2019) | `TRMM/3B42` | GEE | Free | Pre-2014 tropical rainfall climatology baseline | Long rainfall record before GPM |
| 60 | CHIRPS (daily) | UCSB / USGS (USA) | Station + IR-blended | Precipitation | ~5.5 km (0.05°) | Daily | Operational | `UCSB-CHG/CHIRPS/DAILY` | GEE | Free | 40-yr (1981-) gauge-calibrated rainfall — drought indices (SPI), bias reference for IMERG | Long, gauge-bias-corrected rainfall climatology |
| 61 | INSAT rainfall (IMR/GPI/HEM) | ISRO / IMD (India) | Geostationary IR-derived | Precipitation | 4-10 km | 30 min | Operational | External (MOSDAC — not on GEE) | MOSDAC (SAC) | Free* | Sovereign half-hourly rainfall over India; IMD-tuned; fills gauge-sparse interior | Indian geostationary rainfall, IMD-consistent |

---

## 7. THERMAL / EVAPOTRANSPIRATION — Canopy temperature & water use

| # | Name | Operator / Country | Sensor type | Bands / Pol | Native res | Revisit | Status | Cloud asset ID | Portal | Cost | Role in crop / moisture / irrigation system | Gap it fills |
|---|------|--------------------|-------------|-------------|-----------|---------|--------|----------------|--------|------|----------------------------------------------|--------------|
| 62 | ECOSTRESS | NASA-JPL / ISS (USA) | Thermal-IR radiometer | 5 TIR bands (8-12.5 µm) | **70 m** | ~1-5 d (ISS, diurnal sampling) | Operational | `ECO_L2T_LSTE` (LST), `ECO_L3G_JET` (ET) | LP DAAC / **Planetary Computer** | Free* | **70 m land-surface temperature + ET / ESI** — the highest-res spaceborne thermal stress & water-use signal; varied overpass time captures peak-stress hours | High-res, diurnally-sampled thermal stress & ET |
| 63 | Landsat TIRS ST band | USGS-NASA (USA) | Thermal-IR | TIR (100 m → 30 m product) | 30 m (product) | 16 d (8 d L8+L9) | Operational | (in `LANDSAT/LC08(09)/C02/T1_L2`, band `ST_B10`) | GEE | Free | Calibrated 30 m surface temperature for ET (SEBAL/METRIC) & canopy stress | Decametric calibrated thermal time-series |
| 64 | MODIS LST (Terra) | NASA (USA) | Thermal-IR | TIR | 1 km | 8-day composite (daily inst.) | Operational | `MODIS/061/MOD11A2` (LST) | GEE | Free | Daily/8-day 1 km LST for regional thermal stress & heatwave flags | Daily wide-area thermal anchor |
| 65 | MODIS ET | NASA (USA) | ET (PM-based) | ET / PET | 500 m | 8-day | Operational | `MODIS/061/MOD16A2` | GEE | Free | 500 m 8-day actual ET & PET — crop water demand & deficit | Operational regional ET product |
| 66 | VIIRS LST | NOAA-NASA (USA) | Thermal-IR | TIR | 750 m | Daily | Operational | `NASA/VIIRS/002/VNP21A1D` (day LST) | GEE | Free | MODIS-LST continuity; daily thermal into the 2030s | Thermal continuity after MODIS EOL |

---

## 8. DEM — Terrain for hydrology, irrigation slope & SAR/optical correction

| # | Name | Operator / Country | Sensor type | Bands / Pol | Native res | Revisit | Status | Cloud asset ID | Portal | Cost | Role in crop / moisture / irrigation system | Gap it fills |
|---|------|--------------------|-------------|-------------|-----------|---------|--------|----------------|--------|------|----------------------------------------------|--------------|
| 67 | Copernicus DEM GLO-30 | ESA / Airbus (EU) | DEM (TanDEM-X derived) | Elevation | 30 m | Static (2011-2015) | Operational | `COPERNICUS/DEM/GLO30` | GEE | Free | **Primary DEM**: slope/aspect/flow-accumulation for irrigation suitability, terrain-flattening of SAR, orthorectification | Best open global 30 m DEM |
| 68 | SRTM | NASA (USA) | DEM (C-band InSAR) | Elevation | 30 m | Static (2000) | Operational (archive) | `USGS/SRTMGL1_003` | GEE | Free | Legacy DEM baseline; widely-validated hydrology reference | Established 30 m DEM continuity |
| 69 | NASADEM | NASA (USA) | DEM (reprocessed SRTM) | Elevation | 30 m | Static | Operational | `NASA/NASADEM_HGT/001` | GEE | Free | Improved SRTM (voids filled, ICESat-corrected); better hydrology | Cleaner SRTM for flow modelling |
| 70 | ALOS AW3D30 | JAXA (Japan) | DEM (PRISM optical stereo) | Elevation | 30 m | Static | Operational | `JAXA/ALOS/AW3D30/V3_2` | GEE | Free | Independent optical-stereo DEM; cross-check of CopDEM in hilly terrain | DEM independence / verification |
| 71 | ASTER GDEM v3 | NASA / METI (USA/Japan) | DEM (optical stereo) | Elevation | 30 m | Static | Operational (archive) | `NASA/ASTER_GED/AG100_003` (emissivity); GDEM via EarthData | GEE (GED) / Earthdata | Free | Supplementary stereo DEM + surface emissivity for LST retrieval | Emissivity input + DEM gap-fill |

---

## 9. ANCILLARY — Climate reanalysis, land cover & AI embeddings

| # | Name | Operator / Country | Sensor type | Bands / Pol | Native res | Revisit | Status | Cloud asset ID | Portal | Cost | Role in crop / moisture / irrigation system | Gap it fills |
|---|------|--------------------|-------------|-------------|-----------|---------|--------|----------------|--------|------|----------------------------------------------|--------------|
| 72 | ERA5-Land (daily) | ECMWF / C3S (EU) | Reanalysis (model) | Temp, precip, ET, soil-moist layers, radiation | ~9 km (0.1°) | Daily | Operational | `ECMWF/ERA5_LAND/DAILY_AGGR` | GEE | Free | Gap-free met forcing (T, RH, wind, Rn) for ET₀/Penman-Monteith; 4-layer model soil moisture | Continuous, complete weather/energy forcing |
| 73 | AgERA5 | ECMWF / C3S (EU) | Agro-met reanalysis | Ag-tuned met variables | ~10 km | Daily | Operational | External (Copernicus CDS) | Copernicus CDS | Free* | Agriculture-ready daily met (GDD, ET₀) aligned with crop models | Bias-adjusted agro-met variables |
| 74 | NASA POWER | NASA (USA) | Reanalysis (MERRA-2/GEOS) | Solar + met | ~50 km | Daily | Operational | External (POWER API) | power.larc.nasa.gov | Free | Solar radiation & met for ET₀ where ERA5 is coarse; easy point API for advisories | Solar-radiation-focused met API |
| 75 | ESA WorldCover v200 | ESA (EU) | Land-cover map (S1+S2) | 11 classes | 10 m | Annual (2020, 2021) | Operational | `ESA/WorldCover/v200` | GEE | Free | 10 m cropland mask & non-ag exclusion; stratification of analysis | Best open 10 m static land-cover baseline |
| 76 | Dynamic World | Google / WRI (USA) | Near-real-time LULC (S2 + DL) | 9 class probabilities | 10 m | ~5 d (per S2 scene) | Operational | `GOOGLE/DYNAMICWORLD/V1` | GEE | Free | Near-real-time cropland probability; in-season change & fallow/active mapping | Continuous 10 m LULC (vs annual maps) |
| 77 | MODIS Land Cover | NASA (USA) | LULC (multiple schemes) | IGBP/PFT/LAI classes | 500 m | Annual | Operational | `MODIS/061/MCD12Q1` | GEE | Free | Long annual LC record (2001-) for multi-year crop/agro-ecology context | Decadal LULC time-series |
| 78 | AlphaEarth Satellite Embedding | Google DeepMind (USA) | AI embedding (multi-sensor) | **64-D** learned features | **10 m** | Annual (2017-2024) | Operational | `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` | GEE | Free | 64-D per-pixel embedding of S2+S1+L8/9+ERA5+GEDI+GLO30 — few-shot crop classification & similarity search with tiny labels | Sensor-agnostic compressed feature space; few-label crop mapping |

---

## 10. GAP-FILLING MATRIX — Failure mode → compensating assets

The architecture's central principle: **no single sensor is a single point of failure.** Each agronomic/observational failure mode is covered by complementary assets across physics (optical vs radar vs microwave vs thermal) and orbit (LEO vs GEO, polar vs ISS).

| # | Failure mode / limitation | Why it breaks single-sensor systems | Compensating assets (catalog #) |
|---|---------------------------|--------------------------------------|----------------------------------|
| 1 | **Monsoon / kharif cloud cover** (optical blackout Jun-Sep) | NDVI/optical unusable for weeks during peak growth | **SAR cloud-immune:** S1A/C (38,39), **NISAR** (40), EOS-04 (41), ALOS-2 (43), RCM (45), SAOCOM (48); microwave SMAP (52,53); GEO timing to grab clear minutes: Himawari (20), INSAT-3D/R/S (23) |
| 2 | **Coarse soil-moisture footprint** (9-40 km microwave can't resolve a field) | SMAP/SMOS pixels span many farms; no plot-level SM | **SAR downscaling:** S1 SSM 1 km (57), NISAR L-band (40), SAOCOM/EOS-04 (48,41); root-zone anchor SMAP L4 (52); fuse with optical/thermal stress (1,62) for disaggregation |
| 3 | **Low optical revisit** (16 d Landsat / 10 d single-S2 too sparse for stress dynamics) | Miss rapid wilting/irrigation events between passes | S2 A+B+C 3-4 d (1-3), MODIS/VIIRS daily (8-10), PlanetScope daily 3 m (24), Sentinel-3 (11), GEO sub-hourly (20-23); embeddings densify (78) |
| 4 | **Missing thermal band** (S2 has none; canopy-temp stress invisible) | Pre-visual water stress (stomatal closure) undetectable from S2 alone | **ECOSTRESS 70 m** (62), Landsat ST 30 m (63), MODIS LST/ET (64,65), VIIRS LST (66), INSAT/GOES TIR (22,23) |
| 5 | **Terrain shadow & layover** (hills corrupt optical illumination + SAR geometry) | False low-NDVI; SAR foreshortening in undulating/terraced terrain | DEM correction CopDEM GLO-30 (67), NASADEM (69), AW3D30 (70), SRTM (68); SAR terrain-flattening; multi-look-angle SAR (38-48) |
| 6 | **Crop-type confusion** (spectrally similar crops, mixed pixels, smallholders) | NDVI alone can't separate maize/sorghum/cotton; sub-pixel mixing | Red-edge S2/GF-6 (1,19), **hyperspectral** PRISMA/EnMAP/EMIT (33-35), SAR structure/phenology (38-43), AlphaEarth embeddings (78), high-res LISS-IV/PlanetScope (14,24), Dynamic World priors (76) |
| 7 | **Rain-gauge sparsity** (interior India gauge-poor; point gauges miss convective cells) | Water-balance & irrigation-need estimates biased without rainfall | GPM IMERG V07 (58), CHIRPS (60), INSAT rainfall (61), ERA5-Land/AgERA5 (72,73); cross-validate satellite vs gauge |
| 8 | **Single-sensor outage / EOL** (S1B lost 2022; MODIS ageing; mission gaps) | Operational continuity at risk; archive discontinuity | S1C restores SAR (39); VIIRS succeeds MODIS (10); L8+L9 redundancy (4,5); NISAR + EOS-04 + SAOCOM L-band redundancy (40,41,48); SMOS backs SMAP (54); GF/RADARSAT/COSMO independent tasking (18,44,47) |
| 9 | **Sub-meter validation & disputes** (advisory accuracy, insurance, ground-truth) | Coarse pixels can't adjudicate field boundaries / localized events | Cartosat-2/3 (16,17), Pléiades/Neo 0.3 m (28,29), WorldView-2/3 (30,31), GeoEye-1 (32), SkySat (26), ICEYE/Capella SAR (50,51), RISAT-2B (42) |

---

## 11. FUSION TIERS — Operational layering strategy

The system is engineered around a small, free, GEE-native **core** that runs every cycle, a broader **robustness** layer pulled in on demand or for validation, and an **optional commercial** layer for high-value or dispute scenarios.

### Tier-1 — Operational core (12 assets · free · mostly GEE-native)

The minimum viable, always-on fusion stack. Every one is free; all but NISAR (external/ASF) are GEE-native.

| Tier-1 asset | Catalog # | Cloud asset ID | Primary contribution |
|--------------|-----------|----------------|----------------------|
| Sentinel-2 (A/B/C) | 1-3 | `COPERNICUS/S2_SR_HARMONIZED` | 10 m optical indices, crop-type, vigour |
| Sentinel-1 (A/C) | 38,39 | `COPERNICUS/S1_GRD` | All-weather C-band structure & flood |
| Landsat 8 / 9 | 4,5 | `LANDSAT/LC08(09)/C02/T1_L2` | Surface reflectance + 30 m thermal (ST) |
| MODIS (Terra/Aqua) | 8,9 | `MODIS/061/MOD13Q1` | Daily 250 m VI / phenology |
| VIIRS | 10 | `NOAA/VIIRS/001/VNP09GA` | Daily VI continuity (post-MODIS) |
| SMAP (L4 + L3) | 52,53 | `NASA/SMAP/SPL4SMGP/008` | Root-zone + surface soil moisture |
| GPM IMERG V07 | 58 | `NASA/GPM_L3/IMERG_V07` | Half-hourly rainfall forcing |
| CHIRPS | 60 | `UCSB-CHG/CHIRPS/DAILY` | Gauge-calibrated rainfall climatology |
| ECOSTRESS | 62 | `ECO_L2T_LSTE` (Planetary Computer) | 70 m thermal stress + ET |
| **NISAR** | 40 | `NISAR_L1/L2` (ASF DAAC) | **L-band biomass + soil moisture (HEADLINE)** |
| EOS-04 / RISAT-1A | 41 | Bhoonidhi | Sovereign C-band SAR |
| Copernicus DEM GLO-30 | 67 | `COPERNICUS/DEM/GLO30` | Terrain / slope / SAR correction |

### Tier-2 — Robustness & cross-validation (~20 assets · free/free* · mixed GEE + external)

Sentinel-3 OLCI (11), PROBA-V (12), Resourcesat LISS-III/IV/AWiFS (13-15), Cartosat-2/3 (16,17), Himawari (20), GOES-16/18 (21,22), INSAT-3D/R/S (23), PRISMA (33), EnMAP (34), EMIT (35), Hyperion (37), Sentinel-1C (39), ALOS-2 PALSAR-2 (43), RCM (45), SAOCOM (48), SMOS (54), ASCAT (55), AMSR2 (56), Sentinel-1 SSM 1 km (57), TRMM (60-archive), INSAT rainfall (61), Landsat ST (63), MODIS LST/ET (64,65), VIIRS LST (66), SRTM/NASADEM/AW3D30/ASTER (68-71), ERA5-Land/AgERA5/NASA POWER (72-74), ESA WorldCover (75), Dynamic World (76), MODIS LC (77), AlphaEarth embeddings (78).

### Tier-3 — Commercial / optional (high-value, tasked, validation)

PlanetScope (24), RapidEye (25), SkySat (26), SPOT-6/7 (27), Pléiades 1A/1B (28), Pléiades Neo (29), WorldView-2/3 (30,31), GeoEye-1 (32), DESIS (36), RADARSAT-2 (44), TerraSAR-X/TanDEM-X (46), COSMO-SkyMed (47), GaoFen-1/3/6 (18,19,49), ICEYE (50), Capella (51), RISAT-2B (42, tasking modes).

---

## 12. ROLE-BY-PIPELINE-STAGE summary

| Pipeline stage | Lead assets | Supporting assets |
|----------------|-------------|-------------------|
| **Crop-type classification** | S2 red-edge (1-3), AlphaEarth embeddings (78), S1/NISAR structure (38-40) | Hyperspectral (33-35), LISS-IV/PlanetScope (14,24), Dynamic World (76), WorldCover (75) |
| **Moisture-stress detection** | SMAP L4 root-zone (52), ECOSTRESS ET/LST (62), S2 NDMI/NDWI (1-3) | NISAR L-band (40), S1 SSM (57), Landsat ST (63), SMOS/ASCAT/AMSR2 (54-56), thermal MODIS/VIIRS (64,66) |
| **Irrigation advisory** | SMAP root-zone (52) + GPM/CHIRPS rainfall (58,60) + ERA5-Land ET₀ (72) | ECOSTRESS ET (62), MODIS ET (65), CopDEM slope (67), Dynamic World (76), NASA POWER (74) |
| **Validation / ground-truth** | Cartosat-3 (17), Pléiades Neo (29), WorldView-3 (31) | SkySat (26), ICEYE/Capella (50,51), field CCE data |
| **Acquisition planning / cloud timing** | Himawari (20), INSAT-3D/R/S (23), GOES (21,22) | Dynamic World cloud probs, S2 cloud masks |

---

## 13. GEE-NATIVE vs EXTERNAL-INGESTION

A deliberate split: **GEE-native** assets are loaded directly by asset ID inside Earth Engine (zero ingestion cost). **External-ingestion** assets require pulling via Planetary Computer STAC, NASA Earthdata/LP DAAC, ASF DAAC, ISRO Bhoonidhi, ISRO/MOSDAC, JAXA G-Portal, or DLR EOWEB, then harmonising into the analysis grid.

### GEE-native (load by asset ID — no ingestion)

| Asset | ID |
|-------|----|
| Sentinel-2 SR | `COPERNICUS/S2_SR_HARMONIZED` |
| Sentinel-1 GRD | `COPERNICUS/S1_GRD` |
| Landsat 5/7/8/9 L2 | `LANDSAT/LT05(LE07/LC08/LC09)/C02/T1_L2` |
| MODIS VI / SR / LST / ET / LC | `MODIS/061/MOD13Q1`, `MOD09GA`, `MYD13Q1`, `MOD11A2`, `MOD16A2`, `MCD12Q1` |
| VIIRS SR / VI / LST | `NOAA/VIIRS/001/VNP09GA`, `NASA/VIIRS/002/VNP13A1`, `VNP21A1D` |
| Sentinel-3 OLCI | `COPERNICUS/S3/OLCI` |
| PROBA-V 100 m | `VITO/PROBAV/C1/S1_TOC_100M` |
| GOES-16/18 ABI | `NOAA/GOES/16/MCMIPF`, `NOAA/GOES/18/FDCF` |
| ALOS-2 PALSAR-2 yearly mosaic | `JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH` |
| SMAP L4 / L3 | `NASA/SMAP/SPL4SMGP/008`, `NASA/SMAP/SPL3SMP_E/006` |
| GPM IMERG V07 / TRMM | `NASA/GPM_L3/IMERG_V07`, `TRMM/3B42` |
| CHIRPS | `UCSB-CHG/CHIRPS/DAILY` |
| Hyperion (archive) | `EO1/HYPERION` |
| DEMs | `COPERNICUS/DEM/GLO30`, `USGS/SRTMGL1_003`, `NASA/NASADEM_HGT/001`, `JAXA/ALOS/AW3D30/V3_2` |
| ERA5-Land | `ECMWF/ERA5_LAND/DAILY_AGGR` |
| WorldCover / Dynamic World | `ESA/WorldCover/v200`, `GOOGLE/DYNAMICWORLD/V1` |
| AlphaEarth Satellite Embedding | `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` |

### External-ingestion (need ingestion pipeline)

| Asset | Source portal | Note |
|-------|---------------|------|
| **NISAR** (L-band L1/L2) | **ASF DAAC** (+ Bhoonidhi S-band) | HEADLINE; 36-72 h latency; ingest via ASF SearchAPI |
| ECOSTRESS LST/ET | **Planetary Computer** STAC / LP DAAC | `ECO_L2T_LSTE`, `ECO_L3G_JET` |
| EMIT hyperspectral | **Planetary Computer** STAC / LP DAAC | `EMIT_L2A_RFL` |
| PRISMA | ASI / **Bhoonidhi** (India AOIs) | `PRS_L2D` |
| EnMAP / DESIS | **DLR EOWEB** GeoPortal | `ENMAP_HSI_L2A` |
| Himawari-8/9 AHI | **JAXA** Himawari Monitor / P-Tree | Geostationary, not on GEE |
| INSAT-3D/3DR/3DS + INSAT rainfall | **MOSDAC** (SAC/ISRO) | Geostationary + rainfall, not on GEE |
| Resourcesat / Cartosat / EOS-04 / RISAT | **Bhoonidhi** (NRSC) | Sovereign Indian archives |
| SMOS | **CATDS** / ESA | L-band SM |
| ASCAT SWI / Sentinel-1 SSM 1 km | **Copernicus Global Land** / EUMETSAT | Microwave/SAR-derived SM |
| AMSR2 (GCOM-W) | **JAXA G-Portal** / GES DISC (LPRM) | Multi-freq SM + VOD |
| AgERA5 | **Copernicus CDS** | Agro-met |
| NASA POWER | POWER API | Met/solar point service |
| Commercial (Planet / Airbus / Maxar / RADARSAT / TSX / COSMO / SAOCOM / GaoFen / ICEYE / Capella) | Respective vendor APIs | Tasking/commercial |

---

## 14. VERIFICATION NOTES

- **Deprecated SMAP 10 km** — `NASA_USDA/HSL/SMAP10KM_soil_moisture` (and the `NASA_USDA/HSL/SMAP_soil_moisture` 0.25° product) are **deprecated** in the GEE catalog (data 2015-04-02 → 2020-12-31 only). The system uses **`NASA/SMAP/SPL4SMGP/008`** (L4, 9 km, surface + root-zone) and **`NASA/SMAP/SPL3SMP_E/006`** (L3 enhanced, 9 km). Do not reintroduce the deprecated IDs.
- **GPM IMERG V07** — V07 is the current algorithm version; use `NASA/GPM_L3/IMERG_V07` (the older `IMERG_V06`/`IMERG_V05` paths are superseded).
- **NISAR live in 2025** — launched **30 July 2025** aboard GSLV-F16 from Satish Dhawan SC; dual L+S-band; **242 km swath**; **12-day** exact-repeat. As of mid-2026 the mission team has released **100,000+ L1-L3 L-band products** via ASF DAAC (nominal L1-3 latency 36-72 h). S-band products distributed via ISRO Bhoonidhi. This is the system's HEADLINE asset.
- **Sentinel-2C operational** — S2C (launched Sep 2024) is commissioned and took over the S2A operational slot in 2025; all three feed `COPERNICUS/S2_SR_HARMONIZED`.
- **Sentinel-1C operational** — S1C (launched Dec 2024) replaced the lost S1B and restores ~6-day combined C-band cadence; data via `COPERNICUS/S1_GRD`.
- **AlphaEarth Satellite Embedding** — `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, 64 dimensions, 10 m, annual composites **2017-2024**, trained on Landsat 8/9, Sentinel-2, Sentinel-1, GEDI, ERA5-Land, GRACE, GLO-30 and text; ~1.4 trillion footprints/year; near-global excluding > ~82° N/S.
- **Landsat 7** decommissioned (2024) — archive-only; SLC-off striping affects post-2003 scenes.
- **SMAP SPL4SMGP/008** is the current L4 version (supersedes /007); provides 0-5 cm surface and 0-100 cm root-zone soil moisture plus ET/Rn ancillaries.

---

## 15. REFERENCES (source URLs)

**Google Earth Engine Data Catalog**
- Sentinel-2 SR Harmonized — https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
- Sentinel-1 GRD — https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD
- Landsat 9 C02 T1 L2 — https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_L2
- Landsat 8 C02 T1 L2 — https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2
- MODIS MOD13Q1.061 VI — https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13Q1
- VIIRS VNP09GA — https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_001_VNP09GA
- Sentinel-3 OLCI — https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S3_OLCI
- SMAP SPL4SMGP.008 — https://developers.google.com/earth-engine/datasets/catalog/NASA_SMAP_SPL4SMGP_008
- SMAP SPL3SMP_E.006 — https://developers.google.com/earth-engine/datasets/catalog/NASA_SMAP_SPL3SMP_E_006
- NASA-USDA SMAP [deprecated] — https://developers.google.com/earth-engine/datasets/catalog/NASA_USDA_HSL_SMAP_soil_moisture
- GPM IMERG V07 — https://developers.google.com/earth-engine/datasets/catalog/NASA_GPM_L3_IMERG_V07
- CHIRPS Daily — https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY
- MODIS MOD11A2 LST — https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD11A2
- MODIS MOD16A2 ET — https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD16A2
- ALOS PALSAR Yearly Mosaic — https://developers.google.com/earth-engine/datasets/catalog/JAXA_ALOS_PALSAR_YEARLY_SAR_EPOCH
- Copernicus DEM GLO-30 — https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_DEM_GLO30
- SRTM GL1 003 — https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003
- NASADEM HGT 001 — https://developers.google.com/earth-engine/datasets/catalog/NASA_NASADEM_HGT_001
- ALOS AW3D30 V3_2 — https://developers.google.com/earth-engine/datasets/catalog/JAXA_ALOS_AW3D30_V3_2
- ERA5-Land Daily Aggregated — https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_AGGR
- ESA WorldCover v200 — https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200
- Dynamic World V1 — https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1
- MODIS MCD12Q1 Land Cover — https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD12Q1
- Satellite Embedding (AlphaEarth) intro — https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-01-introduction
- Hyperion EO-1 — https://developers.google.com/earth-engine/datasets/catalog/EO1_HYPERION

**NISAR**
- ISRO NISAR mission (GSLV-F16) — https://www.isro.gov.in/Mission_GSLVF16_NISAR.html
- NASA-JPL NISAR — https://www.jpl.nasa.gov/missions/nasa-isro-synthetic-aperture-radar-nisar/
- NISAR Data User Guide (ASF) — https://nisar-docs.asf.alaska.edu/nisar-intro/
- ASF — 100k+ NISAR products released — https://asf.alaska.edu/notices/jpl-releases-over-100k-new-nisar-data-product-files/
- Bhoonidhi NISAR — https://bhoonidhi.nrsc.gov.in/NISAR/
- NISAR L-SAR (Earthdata) — https://www.earthdata.nasa.gov/data/instruments/l-sar

**ISRO / India portals**
- Bhoonidhi (NRSC data portal) — https://bhoonidhi.nrsc.gov.in/
- MOSDAC (SAC — INSAT, oceanic/atmospheric) — https://www.mosdac.gov.in/

**External ingestion sources**
- Microsoft Planetary Computer (STAC: ECOSTRESS, EMIT, etc.) — https://planetarycomputer.microsoft.com/
- ECOSTRESS (LP DAAC) — https://lpdaac.usgs.gov/products/eco_l2t_lstev002/
- EMIT (LP DAAC) — https://lpdaac.usgs.gov/products/emitl2arflv001/
- PRISMA (ASI) — https://www.asi.it/en/earth-science/prisma/
- EnMAP (DLR EOWEB) — https://www.enmap.org/
- Alaska Satellite Facility (ASF DAAC) — https://asf.alaska.edu/
- Copernicus Global Land Service (ASCAT SWI, S1 SSM) — https://land.copernicus.eu/global/
- JAXA G-Portal (AMSR2, ALOS) — https://gportal.jaxa.jp/
- JAXA Himawari Monitor — https://www.eorc.jaxa.jp/ptree/
- ESA SMOS / CATDS — https://www.catds.fr/
- NASA POWER — https://power.larc.nasa.gov/
- Copernicus Climate Data Store (AgERA5, ERA5) — https://cds.climate.copernicus.eu/

**Commercial**
- Planet (PlanetScope/SkySat/RapidEye) — https://www.planet.com/
- Airbus OneAtlas (SPOT/Pléiades/Pléiades Neo) — https://oneatlas.airbus.com/
- Maxar (WorldView/GeoEye) — https://www.maxar.com/

---

*End of catalog. 78 numbered entries across 9 categories (40+ distinct satellites/sensors plus ancillary climate, land-cover, DEM and embedding grids). NISAR is the headline dual-frequency SAR asset; the Tier-1 free/GEE-native core (12 assets) is the always-on operational stack, with the gap-filling matrix guaranteeing no single-sensor point of failure.*
