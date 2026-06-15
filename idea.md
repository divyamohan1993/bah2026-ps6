# bah2026-ps6

PROBLEM STATEMENT 6
AI-Driven Automated Crop type, Moisture Stress Detection and irrigation advisory Across Growth Stages Using Moderate Resolution Spectral Signatures (Optical & Microwave Satellite Data)
Overview
Accurate and timely identification of crop types, detection of moisture stress across phenological stages, and translation of crop water deficit into practical irrigation advisories are critical for precision agriculture, drought monitoring, yield forecasting, climate-resilient farm management, and canal command-area water management. Satellite remote sensing, combining optical observations such as LISS-IV, LISS-III, AWiFS, Sentinel-2, Landsat and MODIS with microwave SAR observations such as EOS-04, Sentinel-1 and upcoming NISAR-SAR data, provides continuous and scalable monitoring support. Optical data capture vegetation vigour and biochemical condition, while SAR observations support all-weather monitoring during cloudy monsoon periods and are sensitive to crop structure and surface moisture.

The proposed methodology is intended as an operational and nationally relevant extension of satellite-based agricultural monitoring. By integrating crop-type mapping, phenology-aware moisture stress assessment and irrigation advisory generation, the solution strengthens the type of intelligence required for programmes such as PMKSY, Digital Agriculture Mission, NMSA and PMFBY. The approach adds a near-real-time water-stress layer that can improve crop monitoring, reduce avoidable irrigation losses, support verifiable crop insurance and prepare users for the effective exploitation of indigenous and international optical and microwave satellite fleets.

This proposal focuses on developing an AI-driven automated methodology for crop type classification, stage-wise phenological mapping, moisture stress detection, crop water deficit estimation using multi-source satellite data fusion, machine learning and deep learning models.

Objective
Develop an automated AI/ML methodology for crop type identification for the current cropped season using multi-temporal spectral signatures from the previous cropped season.
Detect moisture stress dynamics across crop growth stages (Integrate phenology-aware modelling) using optical and / microwave indicators for stage-wise stress interpretation
To develop a satellite-based system to estimate 8-day crop water deficit and generate irrigation advisory maps for canal command areas
Expected Outcomes
Within a 30-hour hackathon, a team can demonstrate a functioning pipeline by using a defined pilot area, pre-identified crop labels, cloud-hosted satellite collections and simplified irrigation-advisory logic
A methodology and user interface for generating automated high-resolution crop type maps, stage-wise moisture stress and irrigation advisory maps.
A dashboard-ready output package consisting of colour-coded stress maps, crop growth-stage interpretation, field or pixel-level irrigation advisory layers and time-series visualization.
A scalable model design that can be extended from a pilot irrigation command area to larger agricultural regions and multiple crop seasons
Datasets Required
Optical satellite data: LISS-III/ AWiFS/ Sentinel-2/ Landsat / MODIS for vegetation vigour, phenology and spectral-index generation.
Microwave SAR data: EOS-04 / Sentinel-1 for all-weather monitoring, backscatter analysis, support during cloud-contaminated kharif periods.
Ancillary and meteorological data: Rainfall, reference evapotranspiration or weather-grid data, command-area boundary, canal command layers, soil and crop coefficient information.
Ground information: Existing ground truth points on crop type information from the study area for model training and validation.
The availability of most satellite sources through national and international portals makes the proposed workflow suitable for rapid prototyping as well as later operational scaling.

Data access and readiness: LISS-III/AWiFS and related ISRO datasets can be sourced from Bhuvan or Bhoonidhi as applicable; Sentinel-1/Sentinel-2, Landsat and MODIS support open-access cloud or portal-based processing.

Existing crop ground information and available software resources reduce start-up time for a 30-hour prototype.

Suggested Tools / Technologies
Programming and geospatial processing: Python, Google Earth Engine, MATLAB, R, GDAL, Rasterio and related geospatial libraries.
Machine learning models: Random Forest and XGBoost for tabular and multi-temporal feature classification.
Deep learning models: LSTM or Temporal CNN for time-series dynamics, and U-Net or related deep learning architectures for spatial classification where data volume permits.
Libraries and frameworks: TensorFlow, PyTorch, Scikit-learn, NumPy, Pandas and geospatial visualization libraries.
Cloud and deployment resources: Google Earth Engine, Google Colab, AWS or ISRO Bhuvan-oriented for scalable processing and output hosting.
The technology is proposed to support rapid development and transparent processing. Cloud platforms such as Google Earth Engine and Google Colab can be used for fast data ingestion, compositing and model execution, while Python, R, MATLAB or similar tools can support advanced analysis, modelling and visualisation.

Expected Workflows / Solutions
The proposed workflow is designed on a pilot command area for a 30 hour hackathon, while the same architecture can later be expanded to additional regions, seasons, crops and benchmark datasets.

Data Pre-processing
Atmospheric correction (optical data), Speckle filtering (SAR – e.g., Refined Lee filter), Temporal compositing (weekly/ fortnightly / monthly)
Feature Extraction
Vegetation indices: NDVI, EVI, NDWI. SAR features: VV, VH polarization, Ratio (VH/VV). Texture features (GLCM), Phenological metrics: Start of season (SOS), Peak growth, Length of growing period
Crop Type Classification
Train supervised ML models using labelled data, multi-temporal classification approach, Generate seasonal crop maps
Moisture Stress Detection
Integrate: NDVI anomalies, NDWI, SAR backscatter sensitivity, Develop stress indices: Vegetation Condition Index (VCI), Soil Moisture Index (SMI), Stage-wise stress classification
Water deficit and Irrigation advisory maps
Participants may adopt any methodology (physical crop water balance, crop coefficient / empirical equations) to estimate weekly (8-day) crop water demand (ETc), quantify water deficit against actual evapotranspiration and rainfall, and translate this deficit into an irrigation status map.
AI/ML Model Development
Combine: Satellite features, Train models: Random Forest / LSTM for temporal dynamics and generate predictive outputs
Validation
Compare with: Ground truth crop data, Accuracy metrics: Overall Accuracy, Kappa coefficient
Output
Crop classification map, Moisture stress maps (stage-wise), Time-series dashboards
Evaluation Parameters
Crop classification accuracy target: greater than 85% where representative training and validation samples are available.
Moisture stress classification should be growth-stage aware and should show logical correspondence with satellite-based indices
The prototype should not only classify crops but also demonstrate that the derived water-deficit and irrigation advisory layers are credible for command-area planning and consistent with available reference information.

Image representing Problem statement
Flow diagram from optical and microwave inputs through feature extraction and AI/ML training to crop maps, validation, and stage-wise crop stress outputs
