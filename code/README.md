# Code – Image-based PM Estimation Methods

This folder contains **Jupyter Notebooks and supporting scripts** implementing selected **image-based air quality estimation methods** reviewed and benchmarked within the AQpictures project.

The code is designed to:
- demonstrate the practical implementation of published methods,
- support reproducibility and comparison,
- act as a starting point for adaptation to other study areas.

## Folder structure
data/
│
├── images/ # Sample outdoor images
├── model input/ # Ready to use datasets for performing the predictive models
└── processing outputs/ # Dedicated folder to store necessary outcome of the models 

code/
│
├── 01-visual_comparison/ #  Interactive viewer to compare two images using a swipe overlay
├── 02-image_feature_extraction/ # Clips the desired areas of the images, divides them in sky and ground section, and averages their RGB channel information
├── 03-physical_based_model/ # Performs the physical based model for ground PM2.5 estimation
├── 04-pm25_rgb_analysis/ # Analyzes the relationship between PM₂.₅ concentration and RGB image channel statistics
├── 05-decision_tree/ # Performs the machine learning model (CART) for ground PM2.5 estimation
└── 06-mlp/ # Performs the deep learning model (MLP) for ground PM2.5 estimation


## Description of components

- **images/**  
  The folder includes captured images from the Milano MeteoGiuliacci webcam that will be used as input for the physical model.

- **model input/**  
  The folder includes two datasets. ML_DL_input is a ready-to-use dataset that has all the inputs for the ML and DL models included**,** therefore the user does not need to merge gathered data anymore. PM25_MI_Marche_ARPA    contains the closest authoritative PM2.5 measurements for the webcam images that are used as inputs for the physical model.

- **processing outputs/**  
  Notebooks implementing statistical or machine-learning models relating image features to particulate matter concentrations.

- **01-visual_comparison/**  
  Tools for comparing image-based PM estimates with authoritative ground-based measurements and complementary satellite products.

- **02-image_feature_extraction/**  
  Integrated, step-by-step examples combining preprocessing, feature extraction, modelling, and evaluation.

- **03-physical_based_model/**  
  Integrated, step-by-step examples combining preprocessing, feature extraction, modelling, and evaluation.

- **04-pm25_rgb_analysis/**  
  Integrated, step-by-step examples combining preprocessing, feature extraction, modelling, and evaluation.

- **05-decision_tree/**  
  Integrated, step-by-step examples combining preprocessing, feature extraction, modelling, and evaluation.

- **06-mlp/**  
  Integrated, step-by-step examples combining preprocessing, feature extraction, modelling, and evaluation.

## Notes

- Notebooks are documented inline to clarify assumptions and parameter choices.
- Methods are implemented using open-source Python libraries only.
- Results are meant for **methodological comparison**, not operational air quality monitoring.
