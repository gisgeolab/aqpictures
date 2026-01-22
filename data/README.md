# Data – Sample Datasets

This folder contains **analysis-ready sample datasets** used to demonstrate and test the image-based PM estimation methods implemented in the AQpictures toolkit.

The data are provided to:
- support reproducibility of the example workflows,
- allow users to explore method behaviour,
- facilitate adaptation to other geographic contexts.

## Folder structure
```
data/
│
├── Images/ # Sample outdoor images
├── ARPA/ # Ground-based measurements
├── MODIS/ # Complementary satellite-derived aerosol optical depth
└── ERA5/ # Model based measurments 
```

## Description of datasets

- **Images/**  
  Webcam images are used as input for image-based PM estimation methods.

- **ARPA/**  
  Authoritative ground-based air quality measurements from the regional monitoring network operated by **ARPA Lombardia** (Agenzia Regionale per la Protezione dell’Ambiente della Lombardia), used for validation and benchmarking.  
  Reference: https://www.arpalombardia.it

- **MODIS/**  
  Satellite-derived aerosol products from the **Moderate Resolution Imaging Spectroradiometer (MODIS)** aboard NASA’s Terra and Aqua satellites, used as complementary reference data (e.g. Aerosol Optical Depth).  
  Reference: https://modis.gsfc.nasa.gov  
  Data access: https://ladsweb.modaps.eosdis.nasa.gov

- **ERA5/**  
  Global atmospheric reanalysis dataset produced by the **European Centre for Medium-Range Weather Forecasts (ECMWF)**, providing hourly atmospheric variables derived from model simulations and data assimilation.  
  Reference: https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5  
  Data access: https://cds.climate.copernicus.eu


## Important notes

- Data are provided for **demonstration and benchmarking purposes only**.
- Spatial and temporal matching between datasets has been performed to ensure comparability.
- Users are encouraged to replace or extend these datasets with local data for replication studies.
