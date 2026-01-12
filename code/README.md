# Code – Image-based PM Estimation Methods

This folder contains **Jupyter Notebooks and supporting scripts** implementing selected **image-based air quality estimation methods** reviewed and benchmarked within the AQpictures project.

The code is designed to:
- demonstrate the practical implementation of published methods,
- support reproducibility and comparison,
- act as a starting point for adaptation to other study areas.

## Folder structure
code/
│
├── preprocessing/ # Image and metadata preprocessing routines
├── feature_extraction/ # Extraction of visual features (e.g. colour, contrast, visibility metrics)
├── modelling/ # PM estimation models and regression / learning approaches
├── evaluation/ # Accuracy assessment and comparison with ground-based and satellite data
└── notebooks/ # End-to-end Jupyter notebooks demonstrating complete workflows


## Description of components

- **preprocessing/**  
  Scripts and notebooks for preparing input images, including resizing, colour-space conversion, and quality checks.

- **feature_extraction/**  
  Implementations of visual feature extraction techniques commonly used in image-based PM estimation (e.g. colour ratios, contrast indices).

- **modelling/**  
  Notebooks implementing statistical or machine-learning models relating image features to particulate matter concentrations.

- **evaluation/**  
  Tools for comparing image-based PM estimates with authoritative ground-based measurements and complementary satellite products.

- **notebooks/**  
  Integrated, step-by-step examples combining preprocessing, feature extraction, modelling, and evaluation.

## Notes

- Notebooks are documented inline to clarify assumptions and parameter choices.
- Methods are implemented using open-source Python libraries only.
- Results are meant for **methodological comparison**, not operational air quality monitoring.
