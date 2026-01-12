# Data – Sample Datasets

This folder contains **analysis-ready sample datasets** used to demonstrate and test the image-based PM estimation methods implemented in the AQpictures toolkit.

The data are provided to:
- support reproducibility of the example workflows,
- allow users to explore method behaviour,
- facilitate adaptation to other geographic contexts.

## Folder structure
data/
│
├── images/ # Sample outdoor images (e.g. webcam or ground-level photographs)
├── ground_pm/ # Ground-based particulate matter measurements
├── satellite/ # Complementary satellite-derived air quality products
├── metadata/ # Image metadata (time, location, acquisition conditions)
└── processed/ # Preprocessed and analysis-ready datasets used by the notebooks


## Description of datasets

- **images/**  
  Representative outdoor images used as input for image-based PM estimation methods.

- **ground_pm/**  
  Authoritative PM measurements from ground monitoring stations used for validation and benchmarking.

- **satellite/**  
  Satellite-based aerosol or air quality products used as complementary reference data.

- **metadata/**  
  Geospatial and temporal metadata associated with the images (e.g. camera location, acquisition time).

- **processed/**  
  Harmonized datasets generated during preprocessing and directly used by the code notebooks.

## Important notes

- Data are provided for **demonstration and benchmarking purposes only**.
- Spatial and temporal matching between datasets has been performed to ensure comparability.
- Users are encouraged to replace or extend these datasets with local data for replication studies.
