# Data – Sample Datasets

This folder contains **analysis-ready sample datasets** used to demonstrate and test the image-based PM estimation methods implemented in the AQpictures toolkit.

The data are provided to:
- support reproducibility of the example workflows,
- allow users to explore method behaviour,
- facilitate adaptation to other geographic contexts.

## Folder structure
data/
│
├── Images/ # Sample outdoor images
├── ARPA/ # Ground-based measurements
├── MODIS/ # Complementary satellite-derived aerosol optical depth
└── ERA5/ # Model based measurments 


## Description of datasets

- **Images/**  
  Representative outdoor images used as input for image-based PM estimation methods.

- **ARPA/**  
  Authoritative ground measurements from ground monitoring stations used for validation and benchmarking.

- **MODIS/**  
  Satellite-based aerosol products used as complementary reference data.

- **ERA5/**  
  Global atmospheric reanalysis dataset produced by ECMWF that provides atmospheric variables.

## Important notes

- Data are provided for **demonstration and benchmarking purposes only**.
- Spatial and temporal matching between datasets has been performed to ensure comparability.
- Users are encouraged to replace or extend these datasets with local data for replication studies.
