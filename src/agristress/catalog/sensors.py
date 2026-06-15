"""Sensor catalog for AgriStress.

A typed, dependency-light registry of the Earth-observation sensors the AgriStress
pipeline can fuse for crop-type mapping, phenology-aware moisture-stress detection and
8-day irrigation advisory (ISRO BAH 2026 PS6).

The module is deliberately built on the standard library only (``dataclasses`` + ``enum``)
so that ``import agristress.catalog.sensors`` is fast and works with *zero* third-party
dependencies or cloud credentials. Heavy clients (Earth Engine, STAC) live in
:mod:`agristress.ingestion`.

Design notes
------------
* Each sensor is described by an immutable :class:`SensorSpec`.
* :data:`SENSOR_REGISTRY` maps a short, stable ``id`` -> :class:`SensorSpec`.
* "Tier" encodes operational priority for the hackathon pilot:

  - **Tier 1** — free/open, cloud-native (GEE/STAC), used by the core pipeline.
  - **Tier 2** — free/open but secondary (gap-fill, archive, specialised).
  - **Tier 3** — commercial / tasking (very-high-res, validation, demos only).

* ``fills_gaps`` lists the *failure modes* a sensor compensates for. The canonical
  failure-mode vocabulary is enumerated in :data:`FAILURE_MODES` and queried via
  :func:`gap_fillers_for`.

Asset identifiers (``gee_asset_id`` / ``stac_collection``) are the real, current
public collection ids on Google Earth Engine and on STAC catalogs (Microsoft Planetary
Computer, ASF, NRSC Bhoonidhi). They are validated structurally — never *fetched* — at
import time, so this module stays offline-safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import StrEnum

__all__ = [
    "FAILURE_MODES",
    "SENSOR_REGISTRY",
    "Cost",
    "SensorSpec",
    "SensorType",
    "by_tier",
    "by_type",
    "gap_fillers_for",
    "gee_native",
    "get_sensor",
    "registry_summary",
    "stac_native",
]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class SensorType(StrEnum):
    """Broad physical family of an EO sensor / product."""

    OPTICAL = "OPTICAL"
    SAR = "SAR"
    RADIOMETER = "RADIOMETER"  # passive microwave (soil-moisture, e.g. SMAP/SMOS)
    THERMAL = "THERMAL"  # thermal-IR / LST / ET
    PRECIP = "PRECIP"  # precipitation products
    DEM = "DEM"  # digital elevation / terrain
    HYPERSPECTRAL = "HYPERSPECTRAL"
    ANCILLARY = "ANCILLARY"  # land-cover, reanalysis, embeddings, etc.


class Cost(StrEnum):
    """Data-access cost / licensing class."""

    FREE = "FREE"  # free, no registration / fully public
    OPEN = "OPEN"  # open but registration / portal account required
    COMMERCIAL = "COMMERCIAL"  # paid / tasking


#: Canonical failure-mode vocabulary used by :func:`gap_fillers_for` and ``fills_gaps``.
FAILURE_MODES: tuple[str, ...] = (
    "monsoon_cloud",  # optical blackout under kharif/monsoon cloud cover
    "coarse_soil_moisture",  # native passive-MW soil moisture too coarse (~9-40 km)
    "low_optical_revisit",  # need denser optical cadence than a single mission gives
    "missing_thermal",  # LST / ET channel unavailable (ECOSTRESS gap, cloud)
    "terrain_shadow",  # slope/aspect correction, layover, hill-shadow in optical/SAR
    "crop_confusion",  # spectrally similar crops need extra structure/texture cues
    "gauge_sparsity",  # sparse rain-gauge network -> need satellite precipitation
    "sensor_outage",  # a primary mission down -> hot-standby of same family
)


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SensorSpec:
    """Immutable description of a single EO sensor / analysis-ready product.

    Attributes
    ----------
    id:
        Short, stable, lowercase identifier (registry key), e.g. ``"sentinel2"``.
    name:
        Human-readable name, e.g. ``"Sentinel-2 MSI (Surface Reflectance)"``.
    operator:
        Owning agency / company (ESA, NASA, ISRO, JAXA, Planet, ...).
    country:
        Operator's country / bloc.
    sensor_type:
        :class:`SensorType` family.
    bands:
        Optical/thermal band names *or* SAR polarizations (kept as a tuple of str).
        For SAR this typically holds ``("VV", "VH")`` etc.; aliased by ``polarizations``.
    native_resolution_m:
        Best native ground sampling distance in metres (None if not applicable).
    revisit_days:
        Nominal revisit at the equator in days (None for one-off / static products).
    swath_km:
        Nominal swath width in km (None if not applicable).
    status:
        Operational status: ``"operational"``, ``"archive"``, ``"planned"``, ``"degraded"``.
    gee_asset_id:
        Google Earth Engine collection id, if the product is GEE-native.
    stac_collection:
        STAC collection id, if reachable via a STAC API (see :mod:`agristress.catalog.assets`).
    portal:
        Primary human access portal / catalog (GEE, Planetary Computer, Bhoonidhi, ...).
    cost:
        :class:`Cost` access class.
    tier:
        Operational tier (1, 2 or 3) — see module docstring.
    roles:
        Pipeline roles this sensor plays, e.g. ``("crop_type", "phenology")``.
    fills_gaps:
        Failure modes (subset of :data:`FAILURE_MODES`) this sensor compensates for.
    """

    id: str
    name: str
    operator: str
    country: str
    sensor_type: SensorType
    bands: tuple[str, ...] = field(default_factory=tuple)
    native_resolution_m: float | None = None
    revisit_days: float | None = None
    swath_km: float | None = None
    status: str = "operational"
    gee_asset_id: str | None = None
    stac_collection: str | None = None
    portal: str = ""
    cost: Cost = Cost.OPEN
    tier: int = 2
    roles: tuple[str, ...] = field(default_factory=tuple)
    fills_gaps: tuple[str, ...] = field(default_factory=tuple)

    # ``polarizations`` is a convenience alias over ``bands`` for SAR sensors.
    @property
    def polarizations(self) -> tuple[str, ...]:
        """Return ``bands`` interpreted as SAR polarizations (empty for non-SAR)."""
        return self.bands if self.sensor_type is SensorType.SAR else ()

    @property
    def is_gee_native(self) -> bool:
        return bool(self.gee_asset_id)

    @property
    def is_stac_native(self) -> bool:
        return bool(self.stac_collection)

    def __post_init__(self) -> None:
        # Validate that declared gap-fills use the canonical vocabulary. We raise here
        # (at construction time) so a typo in the registry is caught the moment this
        # module is imported, rather than silently producing empty query results.
        unknown = set(self.fills_gaps) - set(FAILURE_MODES)
        if unknown:
            raise ValueError(
                f"SensorSpec {self.id!r} declares unknown failure modes {sorted(unknown)}; "
                f"valid modes are {FAILURE_MODES}"
            )
        if self.tier not in (1, 2, 3):
            raise ValueError(f"SensorSpec {self.id!r} has invalid tier {self.tier!r} (want 1/2/3)")


# Names of the dataclass fields that must always be populated for a spec to be "complete".
REQUIRED_FIELDS: tuple[str, ...] = (
    "id",
    "name",
    "operator",
    "country",
    "sensor_type",
    "status",
    "cost",
    "tier",
)


def _spec(**kwargs: object) -> SensorSpec:
    """Tiny constructor helper that coerces list bands/roles/gaps -> tuples."""
    for key in ("bands", "roles", "fills_gaps"):
        val = kwargs.get(key)
        if isinstance(val, list):
            kwargs[key] = tuple(val)
    return SensorSpec(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# fmt: off
_SENSORS: list[SensorSpec] = [
    # ===================================================================
    # TIER 1 — free/open, cloud-native, the operational core of AgriStress
    # ===================================================================
    _spec(
        id="sentinel2", name="Sentinel-2 MSI (Surface Reflectance, harmonized)",
        operator="ESA / Copernicus", country="EU", sensor_type=SensorType.OPTICAL,
        bands=("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"),
        native_resolution_m=10, revisit_days=5, swath_km=290, status="operational",
        gee_asset_id="COPERNICUS/S2_SR_HARMONIZED", stac_collection="sentinel-2-l2a",
        portal="Google Earth Engine / Copernicus DSE / Planetary Computer",
        cost=Cost.FREE, tier=1,
        roles=("crop_type", "phenology", "ndvi", "ndwi", "moisture_stress"),
        fills_gaps=("crop_confusion",),
    ),
    _spec(
        id="sentinel1", name="Sentinel-1 C-band SAR (GRD)",
        operator="ESA / Copernicus", country="EU", sensor_type=SensorType.SAR,
        bands=("VV", "VH"), native_resolution_m=10, revisit_days=6, swath_km=250,
        status="operational", gee_asset_id="COPERNICUS/S1_GRD",
        stac_collection="sentinel-1-grd",
        portal="Google Earth Engine / Copernicus DSE / Planetary Computer",
        cost=Cost.FREE, tier=1,
        roles=("crop_type", "structure", "surface_moisture", "all_weather"),
        fills_gaps=("monsoon_cloud", "crop_confusion", "low_optical_revisit"),
    ),
    _spec(
        id="landsat8", name="Landsat 8 OLI/TIRS (Collection 2, L2 SR/ST)",
        operator="USGS / NASA", country="USA", sensor_type=SensorType.OPTICAL,
        bands=("SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10"),
        native_resolution_m=30, revisit_days=16, swath_km=185, status="operational",
        gee_asset_id="LANDSAT/LC08/C02/T1_L2", stac_collection="landsat-c2-l2",
        portal="Google Earth Engine / USGS EarthExplorer / Planetary Computer",
        cost=Cost.FREE, tier=1,
        roles=("crop_type", "phenology", "thermal", "lst"),
        fills_gaps=("missing_thermal", "low_optical_revisit"),
    ),
    _spec(
        id="landsat9", name="Landsat 9 OLI-2/TIRS-2 (Collection 2, L2 SR/ST)",
        operator="USGS / NASA", country="USA", sensor_type=SensorType.OPTICAL,
        bands=("SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10"),
        native_resolution_m=30, revisit_days=16, swath_km=185, status="operational",
        gee_asset_id="LANDSAT/LC09/C02/T1_L2", stac_collection="landsat-c2-l2",
        portal="Google Earth Engine / USGS EarthExplorer / Planetary Computer",
        cost=Cost.FREE, tier=1,
        roles=("crop_type", "phenology", "thermal", "lst"),
        fills_gaps=("missing_thermal", "low_optical_revisit"),
    ),
    _spec(
        id="modis_terra", name="MODIS Terra Vegetation Indices (MOD13Q1, 16-day 250 m)",
        operator="NASA", country="USA", sensor_type=SensorType.OPTICAL,
        bands=("NDVI", "EVI", "sur_refl_b01", "sur_refl_b02"),
        native_resolution_m=250, revisit_days=16, swath_km=2330, status="operational",
        gee_asset_id="MODIS/061/MOD13Q1",
        portal="Google Earth Engine / LP DAAC", cost=Cost.FREE, tier=1,
        roles=("phenology", "ndvi", "anomaly_baseline"),
        fills_gaps=("monsoon_cloud", "low_optical_revisit"),
    ),
    _spec(
        id="viirs", name="VIIRS/SNPP Vegetation Indices (VNP13A1, 16-day 500 m)",
        operator="NASA / NOAA", country="USA", sensor_type=SensorType.OPTICAL,
        bands=("NDVI", "EVI", "EVI2"),
        native_resolution_m=500, revisit_days=16, swath_km=3060, status="operational",
        gee_asset_id="NASA/VIIRS/002/VNP13A1",
        portal="Google Earth Engine / LP DAAC", cost=Cost.FREE, tier=1,
        roles=("phenology", "ndvi", "continuity"),
        fills_gaps=("low_optical_revisit", "sensor_outage"),
    ),
    _spec(
        id="smap", name="SMAP L4 Global Surface & Root-Zone Soil Moisture (SPL4SMGP)",
        operator="NASA", country="USA", sensor_type=SensorType.RADIOMETER,
        bands=("sm_surface", "sm_rootzone"),
        native_resolution_m=9000, revisit_days=3, swath_km=1000, status="operational",
        gee_asset_id="NASA/SMAP/SPL4SMGP/008",
        portal="Google Earth Engine / NSIDC DAAC", cost=Cost.FREE, tier=1,
        roles=("soil_moisture", "stress_index", "smi"),
        fills_gaps=("coarse_soil_moisture",),
    ),
    _spec(
        id="gpm_imerg", name="GPM IMERG V07 Half-Hourly Precipitation",
        operator="NASA / JAXA", country="USA/Japan", sensor_type=SensorType.PRECIP,
        bands=("precipitation", "precipitationCal"),
        native_resolution_m=11000, revisit_days=0.02, swath_km=None, status="operational",
        gee_asset_id="NASA/GPM_L3/IMERG_V07",
        portal="Google Earth Engine / GES DISC", cost=Cost.FREE, tier=1,
        roles=("rainfall", "water_balance", "etc_input"),
        fills_gaps=("gauge_sparsity",),
    ),
    _spec(
        id="chirps", name="CHIRPS Daily Precipitation (0.05 deg)",
        operator="UCSB Climate Hazards Center / USGS", country="USA",
        sensor_type=SensorType.PRECIP, bands=("precipitation",),
        native_resolution_m=5566, revisit_days=1, swath_km=None, status="operational",
        gee_asset_id="UCSB-CHG/CHIRPS/DAILY",
        portal="Google Earth Engine / CHC", cost=Cost.FREE, tier=1,
        roles=("rainfall", "water_balance", "drought_baseline"),
        fills_gaps=("gauge_sparsity",),
    ),
    _spec(
        id="ecostress", name="ECOSTRESS L2T Land Surface Temperature & Emissivity",
        operator="NASA / JPL", country="USA", sensor_type=SensorType.THERMAL,
        bands=("LST", "EmisWB"),
        native_resolution_m=70, revisit_days=4, swath_km=384, status="operational",
        stac_collection="eco-l2t-lste",
        portal="Planetary Computer / LP DAAC", cost=Cost.OPEN, tier=1,
        roles=("thermal", "lst", "et", "water_deficit"),
        fills_gaps=("missing_thermal",),
    ),
    _spec(
        id="nisar", name="NISAR L+S-band SAR (dual-frequency)",
        operator="NASA / ISRO", country="USA/India", sensor_type=SensorType.SAR,
        bands=("HH", "HV", "VH", "VV"), native_resolution_m=10, revisit_days=12,
        swath_km=240, status="planned",
        stac_collection="nisar-l-band",
        portal="ASF DAAC (NASA Alaska Satellite Facility)", cost=Cost.OPEN, tier=1,
        roles=("crop_type", "structure", "soil_moisture", "all_weather"),
        fills_gaps=("monsoon_cloud", "crop_confusion", "coarse_soil_moisture"),
    ),
    _spec(
        id="copernicus_dem", name="Copernicus DEM GLO-30 (30 m global)",
        operator="ESA / Airbus", country="EU", sensor_type=SensorType.DEM,
        bands=("DEM",), native_resolution_m=30, revisit_days=None, swath_km=None,
        status="operational", gee_asset_id="COPERNICUS/DEM/GLO30",
        stac_collection="cop-dem-glo-30",
        portal="Google Earth Engine / Planetary Computer", cost=Cost.FREE, tier=1,
        roles=("terrain", "slope", "aspect", "topographic_correction"),
        fills_gaps=("terrain_shadow",),
    ),

    # ===================================================================
    # TIER 2 — free/open, secondary (archive, gap-fill, specialised)
    # ===================================================================
    _spec(
        id="landsat5", name="Landsat 5 TM (Collection 2, L2 SR/ST) — archive",
        operator="USGS / NASA", country="USA", sensor_type=SensorType.OPTICAL,
        bands=("SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7", "ST_B6"),
        native_resolution_m=30, revisit_days=16, swath_km=185, status="archive",
        gee_asset_id="LANDSAT/LT05/C02/T1_L2", stac_collection="landsat-c2-l2",
        portal="Google Earth Engine / USGS EarthExplorer", cost=Cost.FREE, tier=2,
        roles=("historical_baseline", "phenology"),
        fills_gaps=("low_optical_revisit",),
    ),
    _spec(
        id="landsat7", name="Landsat 7 ETM+ (Collection 2, L2 SR/ST) — archive (SLC-off)",
        operator="USGS / NASA", country="USA", sensor_type=SensorType.OPTICAL,
        bands=("SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7", "ST_B6"),
        native_resolution_m=30, revisit_days=16, swath_km=185, status="archive",
        gee_asset_id="LANDSAT/LE07/C02/T1_L2", stac_collection="landsat-c2-l2",
        portal="Google Earth Engine / USGS EarthExplorer", cost=Cost.FREE, tier=2,
        roles=("historical_baseline", "phenology"),
        fills_gaps=("low_optical_revisit",),
    ),
    _spec(
        id="modis_aqua", name="MODIS Aqua Vegetation Indices (MYD13Q1, 16-day 250 m)",
        operator="NASA", country="USA", sensor_type=SensorType.OPTICAL,
        bands=("NDVI", "EVI"), native_resolution_m=250, revisit_days=16, swath_km=2330,
        status="operational", gee_asset_id="MODIS/061/MYD13Q1",
        portal="Google Earth Engine / LP DAAC", cost=Cost.FREE, tier=2,
        roles=("phenology", "ndvi", "afternoon_overpass"),
        fills_gaps=("monsoon_cloud", "low_optical_revisit", "sensor_outage"),
    ),
    _spec(
        id="mod11a2", name="MODIS Terra Land Surface Temperature (MOD11A2, 8-day 1 km)",
        operator="NASA", country="USA", sensor_type=SensorType.THERMAL,
        bands=("LST_Day_1km", "LST_Night_1km"), native_resolution_m=1000,
        revisit_days=8, swath_km=2330, status="operational",
        gee_asset_id="MODIS/061/MOD11A2",
        portal="Google Earth Engine / LP DAAC", cost=Cost.FREE, tier=2,
        roles=("thermal", "lst", "stress_index"),
        fills_gaps=("missing_thermal",),
    ),
    _spec(
        id="mod16a2", name="MODIS Terra Evapotranspiration (MOD16A2, 8-day 500 m)",
        operator="NASA", country="USA", sensor_type=SensorType.THERMAL,
        bands=("ET", "PET"), native_resolution_m=500, revisit_days=8, swath_km=2330,
        status="operational", gee_asset_id="MODIS/061/MOD16A2",
        portal="Google Earth Engine / LP DAAC", cost=Cost.FREE, tier=2,
        roles=("et", "water_deficit", "etc_reference"),
        fills_gaps=("missing_thermal",),
    ),
    _spec(
        id="sentinel3_olci", name="Sentinel-3 OLCI (Ocean & Land Colour, 300 m)",
        operator="ESA / Copernicus", country="EU", sensor_type=SensorType.OPTICAL,
        bands=("Oa08_radiance", "Oa17_radiance"), native_resolution_m=300,
        revisit_days=2, swath_km=1270, status="operational",
        gee_asset_id="COPERNICUS/S3/OLCI",
        portal="Google Earth Engine / Copernicus DSE", cost=Cost.FREE, tier=2,
        roles=("phenology", "ndvi", "wide_swath"),
        fills_gaps=("low_optical_revisit",),
    ),
    _spec(
        id="resourcesat_liss3", name="Resourcesat-2/2A LISS-III (23.5 m)",
        operator="ISRO / NRSC", country="India", sensor_type=SensorType.OPTICAL,
        bands=("B2", "B3", "B4", "B5"), native_resolution_m=23.5, revisit_days=24,
        swath_km=141, status="operational",
        portal="NRSC Bhoonidhi / Bhuvan", cost=Cost.OPEN, tier=2,
        roles=("crop_type", "phenology", "india_priority"),
        fills_gaps=("crop_confusion",),
    ),
    _spec(
        id="resourcesat_awifs", name="Resourcesat-2/2A AWiFS (56 m, wide swath)",
        operator="ISRO / NRSC", country="India", sensor_type=SensorType.OPTICAL,
        bands=("B2", "B3", "B4", "B5"), native_resolution_m=56, revisit_days=5,
        swath_km=740, status="operational",
        portal="NRSC Bhoonidhi / Bhuvan", cost=Cost.OPEN, tier=2,
        roles=("crop_type", "regional_mapping", "india_priority"),
        fills_gaps=("low_optical_revisit", "crop_confusion"),
    ),
    _spec(
        id="insat3d", name="INSAT-3D/3DR Imager (geostationary, thermal/VIS)",
        operator="ISRO / MOSDAC", country="India", sensor_type=SensorType.THERMAL,
        bands=("TIR1", "TIR2", "WV", "VIS"), native_resolution_m=4000,
        revisit_days=0.0104, swath_km=None, status="operational",
        portal="MOSDAC (ISRO)", cost=Cost.OPEN, tier=2,
        roles=("thermal", "lst", "rainfall_proxy", "diurnal"),
        fills_gaps=("missing_thermal", "gauge_sparsity"),
    ),
    _spec(
        id="goes18", name="GOES-18 ABI Full-Disk (geostationary)",
        operator="NOAA", country="USA", sensor_type=SensorType.THERMAL,
        bands=("CMI_C02", "CMI_C13", "Area", "Temp"), native_resolution_m=2000,
        revisit_days=0.0069, swath_km=None, status="operational",
        gee_asset_id="NOAA/GOES/18/FDCF",
        portal="Google Earth Engine / NOAA", cost=Cost.FREE, tier=2,
        roles=("thermal", "fire", "diurnal", "cloud_screening"),
        fills_gaps=("missing_thermal",),
    ),
    _spec(
        id="alos2_palsar", name="ALOS-2 PALSAR-2 L-band SAR (yearly mosaic)",
        operator="JAXA", country="Japan", sensor_type=SensorType.SAR,
        bands=("HH", "HV"), native_resolution_m=25, revisit_days=14, swath_km=70,
        status="operational", gee_asset_id="JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH",
        portal="Google Earth Engine / JAXA", cost=Cost.FREE, tier=2,
        roles=("structure", "biomass", "all_weather", "l_band"),
        fills_gaps=("monsoon_cloud", "crop_confusion"),
    ),
    _spec(
        id="radarsat_rcm", name="RADARSAT Constellation Mission C-band SAR",
        operator="CSA / MDA", country="Canada", sensor_type=SensorType.SAR,
        bands=("HH", "HV", "VV", "VH"), native_resolution_m=3, revisit_days=4,
        swath_km=350, status="operational",
        portal="CSA EODMS", cost=Cost.OPEN, tier=2,
        roles=("structure", "surface_moisture", "all_weather"),
        fills_gaps=("monsoon_cloud", "sensor_outage"),
    ),
    _spec(
        id="saocom", name="SAOCOM-1A/1B L-band SAR",
        operator="CONAE", country="Argentina", sensor_type=SensorType.SAR,
        bands=("HH", "HV", "VH", "VV"), native_resolution_m=10, revisit_days=8,
        swath_km=320, status="operational",
        portal="CONAE", cost=Cost.OPEN, tier=2,
        roles=("soil_moisture", "structure", "l_band", "all_weather"),
        fills_gaps=("monsoon_cloud", "coarse_soil_moisture"),
    ),
    _spec(
        id="smos", name="SMOS L-band Radiometer Soil Moisture (MIRAS)",
        operator="ESA", country="EU", sensor_type=SensorType.RADIOMETER,
        bands=("Soil_Moisture",), native_resolution_m=43000, revisit_days=3,
        swath_km=1000, status="operational",
        portal="ESA / CATDS", cost=Cost.OPEN, tier=2,
        roles=("soil_moisture", "smi", "stress_index"),
        fills_gaps=("coarse_soil_moisture",),
    ),
    _spec(
        id="ascat", name="MetOp ASCAT C-band Scatterometer Soil Moisture",
        operator="EUMETSAT", country="EU", sensor_type=SensorType.RADIOMETER,
        bands=("soil_moisture",), native_resolution_m=25000, revisit_days=1,
        swath_km=1100, status="operational",
        portal="EUMETSAT / Copernicus", cost=Cost.OPEN, tier=2,
        roles=("soil_moisture", "smi"),
        fills_gaps=("coarse_soil_moisture",),
    ),
    _spec(
        id="amsr2", name="GCOM-W AMSR2 Passive Microwave (soil moisture / LST)",
        operator="JAXA", country="Japan", sensor_type=SensorType.RADIOMETER,
        bands=("soil_moisture", "LST"), native_resolution_m=10000, revisit_days=1,
        swath_km=1450, status="operational",
        portal="JAXA G-Portal", cost=Cost.OPEN, tier=2,
        roles=("soil_moisture", "smi"),
        fills_gaps=("coarse_soil_moisture",),
    ),
    _spec(
        id="eos04", name="EOS-04 / RISAT-1A C-band SAR",
        operator="ISRO / NRSC", country="India", sensor_type=SensorType.SAR,
        bands=("HH", "HV", "VV", "VH"), native_resolution_m=25, revisit_days=12,
        swath_km=225, status="operational",
        portal="NRSC Bhoonidhi", cost=Cost.OPEN, tier=2,
        roles=("crop_type", "structure", "surface_moisture", "all_weather",
               "india_priority"),
        fills_gaps=("monsoon_cloud", "crop_confusion", "coarse_soil_moisture"),
    ),
    _spec(
        id="risat", name="RISAT-2B/2BR1 X/C-band SAR",
        operator="ISRO / NRSC", country="India", sensor_type=SensorType.SAR,
        bands=("HH", "HV", "VV", "VH"), native_resolution_m=1, revisit_days=12,
        swath_km=10, status="operational",
        portal="NRSC Bhoonidhi", cost=Cost.OPEN, tier=2,
        roles=("structure", "all_weather", "india_priority"),
        fills_gaps=("monsoon_cloud", "sensor_outage"),
    ),
    _spec(
        id="prisma", name="PRISMA Hyperspectral Imager (240 bands)",
        operator="ASI", country="Italy", sensor_type=SensorType.HYPERSPECTRAL,
        bands=("VNIR", "SWIR"), native_resolution_m=30, revisit_days=29, swath_km=30,
        status="operational",
        portal="ASI PRISMA portal", cost=Cost.OPEN, tier=2,
        roles=("crop_type", "biochemistry", "stress_biochemistry"),
        fills_gaps=("crop_confusion",),
    ),
    _spec(
        id="enmap", name="EnMAP Hyperspectral Imager (224 bands)",
        operator="DLR", country="Germany", sensor_type=SensorType.HYPERSPECTRAL,
        bands=("VNIR", "SWIR"), native_resolution_m=30, revisit_days=27, swath_km=30,
        status="operational",
        portal="DLR EOWEB / Planetary Computer", cost=Cost.OPEN, tier=2,
        roles=("crop_type", "biochemistry", "stress_biochemistry"),
        fills_gaps=("crop_confusion",),
    ),
    _spec(
        id="emit", name="EMIT Imaging Spectrometer (ISS, VSWIR)",
        operator="NASA / JPL", country="USA", sensor_type=SensorType.HYPERSPECTRAL,
        bands=("reflectance",), native_resolution_m=60, revisit_days=None, swath_km=75,
        status="operational", stac_collection="emit-l2a-rfl",
        portal="Planetary Computer / LP DAAC", cost=Cost.OPEN, tier=2,
        roles=("biochemistry", "stress_biochemistry", "mineralogy"),
        fills_gaps=("crop_confusion",),
    ),
    _spec(
        id="hyperion", name="EO-1 Hyperion Hyperspectral (220 bands) — archive",
        operator="NASA / USGS", country="USA", sensor_type=SensorType.HYPERSPECTRAL,
        bands=("VNIR", "SWIR"), native_resolution_m=30, revisit_days=200, swath_km=7.7,
        status="archive", gee_asset_id="EO1/HYPERION",
        portal="Google Earth Engine / USGS", cost=Cost.FREE, tier=2,
        roles=("biochemistry", "historical_baseline"),
        fills_gaps=("crop_confusion",),
    ),
    _spec(
        id="srtm", name="SRTM 30 m DEM (SRTMGL1 v003)",
        operator="NASA / NGA", country="USA", sensor_type=SensorType.DEM,
        bands=("elevation",), native_resolution_m=30, revisit_days=None, swath_km=None,
        status="operational", gee_asset_id="USGS/SRTMGL1_003",
        stac_collection="cop-dem-glo-30",
        portal="Google Earth Engine / LP DAAC", cost=Cost.FREE, tier=2,
        roles=("terrain", "slope", "aspect", "topographic_correction"),
        fills_gaps=("terrain_shadow",),
    ),
    _spec(
        id="nasadem", name="NASADEM 30 m (SRTM reprocessing, HGT v001)",
        operator="NASA", country="USA", sensor_type=SensorType.DEM,
        bands=("elevation",), native_resolution_m=30, revisit_days=None, swath_km=None,
        status="operational", gee_asset_id="NASA/NASADEM_HGT/001",
        stac_collection="nasadem",
        portal="Google Earth Engine / LP DAAC / Planetary Computer",
        cost=Cost.FREE, tier=2,
        roles=("terrain", "slope", "aspect", "topographic_correction"),
        fills_gaps=("terrain_shadow",),
    ),
    _spec(
        id="aw3d30", name="ALOS World 3D 30 m DSM (AW3D30 v3.2)",
        operator="JAXA", country="Japan", sensor_type=SensorType.DEM,
        bands=("DSM",), native_resolution_m=30, revisit_days=None, swath_km=None,
        status="operational", gee_asset_id="JAXA/ALOS/AW3D30/V3_2",
        portal="Google Earth Engine / JAXA", cost=Cost.FREE, tier=2,
        roles=("terrain", "slope", "aspect", "topographic_correction"),
        fills_gaps=("terrain_shadow",),
    ),
    _spec(
        id="era5_land", name="ERA5-Land Daily Aggregated Reanalysis",
        operator="ECMWF / Copernicus", country="EU", sensor_type=SensorType.ANCILLARY,
        bands=("temperature_2m", "total_precipitation_sum",
               "potential_evaporation_sum", "volumetric_soil_water_layer_1"),
        native_resolution_m=11132, revisit_days=1, swath_km=None, status="operational",
        gee_asset_id="ECMWF/ERA5_LAND/DAILY_AGGR",
        portal="Google Earth Engine / Copernicus CDS", cost=Cost.FREE, tier=2,
        roles=("weather", "et0", "water_balance", "etc_input"),
        fills_gaps=("gauge_sparsity", "missing_thermal"),
    ),
    _spec(
        id="esa_worldcover", name="ESA WorldCover 10 m Land Cover (v200)",
        operator="ESA", country="EU", sensor_type=SensorType.ANCILLARY,
        bands=("Map",), native_resolution_m=10, revisit_days=None, swath_km=None,
        status="operational", gee_asset_id="ESA/WorldCover/v200",
        stac_collection="esa-worldcover",
        portal="Google Earth Engine / Planetary Computer", cost=Cost.FREE, tier=2,
        roles=("land_cover", "cropland_mask", "stratification"),
        fills_gaps=("crop_confusion",),
    ),
    _spec(
        id="dynamic_world", name="Dynamic World V1 Near-Real-Time Land Cover (10 m)",
        operator="Google / WRI", country="USA", sensor_type=SensorType.ANCILLARY,
        bands=("crops", "label", "grass", "trees", "water"),
        native_resolution_m=10, revisit_days=5, swath_km=290, status="operational",
        gee_asset_id="GOOGLE/DYNAMICWORLD/V1",
        portal="Google Earth Engine", cost=Cost.FREE, tier=2,
        roles=("land_cover", "cropland_mask", "nrt_dynamics"),
        fills_gaps=("crop_confusion",),
    ),
    _spec(
        id="alphaearth", name="AlphaEarth Foundations Satellite Embedding V1 (annual 10 m)",
        operator="Google DeepMind", country="USA", sensor_type=SensorType.ANCILLARY,
        bands=tuple(f"A{i:02d}" for i in range(64)), native_resolution_m=10,
        revisit_days=365, swath_km=None, status="operational",
        gee_asset_id="GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
        portal="Google Earth Engine", cost=Cost.FREE, tier=2,
        roles=("embedding", "crop_type", "feature_compression", "few_shot"),
        fills_gaps=("crop_confusion", "low_optical_revisit"),
    ),

    # ===================================================================
    # TIER 3 — commercial / tasking (very-high-res, validation, demos)
    # ===================================================================
    _spec(
        id="planetscope", name="PlanetScope (Dove) 3 m daily optical",
        operator="Planet Labs", country="USA", sensor_type=SensorType.OPTICAL,
        bands=("blue", "green", "red", "nir"), native_resolution_m=3, revisit_days=1,
        swath_km=24, status="operational",
        portal="Planet Platform (tasking/archive)", cost=Cost.COMMERCIAL, tier=3,
        roles=("crop_type", "phenology", "field_validation", "daily_optical"),
        fills_gaps=("low_optical_revisit", "monsoon_cloud", "crop_confusion"),
    ),
    _spec(
        id="skysat", name="SkySat 0.5 m optical / video",
        operator="Planet Labs", country="USA", sensor_type=SensorType.OPTICAL,
        bands=("blue", "green", "red", "nir", "pan"), native_resolution_m=0.5,
        revisit_days=1, swath_km=8, status="operational",
        portal="Planet Platform (tasking)", cost=Cost.COMMERCIAL, tier=3,
        roles=("field_validation", "vhr_reference"),
        fills_gaps=("crop_confusion",),
    ),
    _spec(
        id="pleiades", name="Pleiades / Pleiades Neo 0.3-0.5 m optical",
        operator="Airbus", country="France", sensor_type=SensorType.OPTICAL,
        bands=("blue", "green", "red", "nir", "pan"), native_resolution_m=0.3,
        revisit_days=1, swath_km=20, status="operational",
        portal="Airbus OneAtlas (tasking)", cost=Cost.COMMERCIAL, tier=3,
        roles=("field_validation", "vhr_reference"),
        fills_gaps=("crop_confusion",),
    ),
    _spec(
        id="worldview", name="WorldView-2/3 0.3 m optical (8/16 band)",
        operator="Maxar", country="USA", sensor_type=SensorType.OPTICAL,
        bands=("coastal", "blue", "green", "yellow", "red", "rededge", "nir1", "nir2"),
        native_resolution_m=0.31, revisit_days=1, swath_km=13, status="operational",
        portal="Maxar SecureWatch (tasking)", cost=Cost.COMMERCIAL, tier=3,
        roles=("field_validation", "vhr_reference", "stress_biochemistry"),
        fills_gaps=("crop_confusion",),
    ),
    _spec(
        id="spot", name="SPOT-6/7 1.5 m optical",
        operator="Airbus", country="France", sensor_type=SensorType.OPTICAL,
        bands=("blue", "green", "red", "nir", "pan"), native_resolution_m=1.5,
        revisit_days=1, swath_km=60, status="operational",
        portal="Airbus OneAtlas (tasking)", cost=Cost.COMMERCIAL, tier=3,
        roles=("field_validation", "vhr_reference", "crop_type"),
        fills_gaps=("crop_confusion", "low_optical_revisit"),
    ),
    _spec(
        id="terrasar_x", name="TerraSAR-X / TanDEM-X X-band SAR",
        operator="DLR / Airbus", country="Germany", sensor_type=SensorType.SAR,
        bands=("HH", "VV", "HV", "VH"), native_resolution_m=1, revisit_days=11,
        swath_km=30, status="operational",
        portal="Airbus (tasking)", cost=Cost.COMMERCIAL, tier=3,
        roles=("structure", "all_weather", "vhr_sar"),
        fills_gaps=("monsoon_cloud",),
    ),
    _spec(
        id="cosmo_skymed", name="COSMO-SkyMed (1st/2nd gen) X-band SAR",
        operator="ASI / e-GEOS", country="Italy", sensor_type=SensorType.SAR,
        bands=("HH", "VV", "HV", "VH"), native_resolution_m=1, revisit_days=1,
        swath_km=40, status="operational",
        portal="e-GEOS (tasking)", cost=Cost.COMMERCIAL, tier=3,
        roles=("structure", "all_weather", "vhr_sar"),
        fills_gaps=("monsoon_cloud", "sensor_outage"),
    ),
    _spec(
        id="iceye", name="ICEYE X-band SAR (smallsat constellation)",
        operator="ICEYE", country="Finland", sensor_type=SensorType.SAR,
        bands=("VV",), native_resolution_m=0.25, revisit_days=0.5, swath_km=30,
        status="operational",
        portal="ICEYE (tasking)", cost=Cost.COMMERCIAL, tier=3,
        roles=("structure", "all_weather", "rapid_revisit"),
        fills_gaps=("monsoon_cloud", "sensor_outage"),
    ),
    _spec(
        id="capella", name="Capella Space X-band SAR",
        operator="Capella Space", country="USA", sensor_type=SensorType.SAR,
        bands=("HH", "VV"), native_resolution_m=0.5, revisit_days=1, swath_km=10,
        status="operational",
        portal="Capella Console (tasking)", cost=Cost.COMMERCIAL, tier=3,
        roles=("structure", "all_weather", "vhr_sar"),
        fills_gaps=("monsoon_cloud", "sensor_outage"),
    ),
    _spec(
        id="gaofen", name="GaoFen-1/3/6 optical & C-band SAR series",
        operator="CNSA", country="China", sensor_type=SensorType.OPTICAL,
        bands=("blue", "green", "red", "nir"), native_resolution_m=2, revisit_days=4,
        swath_km=90, status="operational",
        portal="CNSA / CRESDA", cost=Cost.COMMERCIAL, tier=3,
        roles=("crop_type", "field_validation"),
        fills_gaps=("crop_confusion", "low_optical_revisit"),
    ),
]
# fmt: on


#: Primary registry: stable ``id`` -> :class:`SensorSpec`.
SENSOR_REGISTRY: dict[str, SensorSpec] = {s.id: s for s in _SENSORS}

# Guard against accidental duplicate ids in the table above.
if len(SENSOR_REGISTRY) != len(_SENSORS):
    _seen: set[str] = set()
    _dupes = sorted({s.id for s in _SENSORS if s.id in _seen or _seen.add(s.id)})  # type: ignore[func-returns-value]
    raise RuntimeError(f"Duplicate sensor ids in SENSOR_REGISTRY: {_dupes}")


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------
def get_sensor(sensor_id: str) -> SensorSpec:
    """Return the :class:`SensorSpec` registered under ``sensor_id``.

    Raises
    ------
    KeyError
        If ``sensor_id`` is not in :data:`SENSOR_REGISTRY` (the error lists close
        matches to help with typos).
    """
    try:
        return SENSOR_REGISTRY[sensor_id]
    except KeyError:
        near = sorted(k for k in SENSOR_REGISTRY if sensor_id.lower() in k.lower())
        hint = f" Did you mean one of {near}?" if near else ""
        raise KeyError(f"Unknown sensor id {sensor_id!r}.{hint}") from None


def by_tier(tier: int) -> list[SensorSpec]:
    """All sensors in the given operational tier (1/2/3), sorted by id."""
    return sorted((s for s in SENSOR_REGISTRY.values() if s.tier == tier), key=lambda s: s.id)


def by_type(sensor_type: SensorType | str) -> list[SensorSpec]:
    """All sensors of a given :class:`SensorType` (accepts the enum or its name)."""
    st = SensorType(sensor_type) if not isinstance(sensor_type, SensorType) else sensor_type
    return sorted((s for s in SENSOR_REGISTRY.values() if s.sensor_type is st), key=lambda s: s.id)


def gee_native() -> list[SensorSpec]:
    """All sensors that expose a Google Earth Engine ``gee_asset_id``, sorted by id."""
    return sorted((s for s in SENSOR_REGISTRY.values() if s.gee_asset_id), key=lambda s: s.id)


def stac_native() -> list[SensorSpec]:
    """All sensors reachable via a STAC ``stac_collection``, sorted by id."""
    return sorted((s for s in SENSOR_REGISTRY.values() if s.stac_collection), key=lambda s: s.id)


def gap_fillers_for(failure_mode: str) -> list[SensorSpec]:
    """Sensors that compensate for a given ``failure_mode``.

    ``failure_mode`` must be one of :data:`FAILURE_MODES`. Results are sorted so the
    cheapest / highest-priority sensors come first (Tier 1 before Tier 2/3, then FREE
    before OPEN before COMMERCIAL, then by id) — i.e. the order you would actually try
    them in an operational fallback chain.
    """
    if failure_mode not in FAILURE_MODES:
        raise ValueError(f"Unknown failure_mode {failure_mode!r}; valid modes are {FAILURE_MODES}")
    _cost_rank = {Cost.FREE: 0, Cost.OPEN: 1, Cost.COMMERCIAL: 2}
    matches = [s for s in SENSOR_REGISTRY.values() if failure_mode in s.fills_gaps]
    return sorted(matches, key=lambda s: (s.tier, _cost_rank[s.cost], s.id))


def registry_summary() -> dict[str, object]:
    """Return a compact, JSON-friendly summary of the registry (handy for CLIs/tests)."""
    return {
        "total": len(SENSOR_REGISTRY),
        "by_tier": {t: len(by_tier(t)) for t in (1, 2, 3)},
        "by_type": {st.value: len(by_type(st)) for st in SensorType},
        "gee_native": len(gee_native()),
        "stac_native": len(stac_native()),
        "failure_modes": {fm: len(gap_fillers_for(fm)) for fm in FAILURE_MODES},
    }


def _validate_registry() -> None:
    """Assert every spec has all :data:`REQUIRED_FIELDS` populated (import-time check)."""
    field_names = {f.name for f in fields(SensorSpec)}
    missing_required = set(REQUIRED_FIELDS) - field_names
    if missing_required:  # pragma: no cover - guards against future refactors
        raise RuntimeError(f"REQUIRED_FIELDS references unknown fields: {missing_required}")
    for spec in SENSOR_REGISTRY.values():
        for fname in REQUIRED_FIELDS:
            value = getattr(spec, fname)
            if value is None or (isinstance(value, str) and not value.strip()):
                raise RuntimeError(f"Sensor {spec.id!r} is missing required field {fname!r}")


_validate_registry()
