# Code – Image-based PM Estimation Methods

This folder contains **Jupyter Notebooks and supporting scripts** implementing selected **image-based air quality estimation methods** reviewed and benchmarked within the AQpictures project.

The code is designed to:
- demonstrate the practical implementation of published methods,
- support reproducibility and comparison,
- act as a starting point for adaptation to other study areas.

## Folder structure
```
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
```

## Description of components

- **images/**  
  The folder includes captured images from the Milano MeteoGiuliacci webcam that will be used as input for the physical model.

- **model input/**  
  The folder includes two datasets. ML_DL_input is a ready-to-use dataset that has all the inputs for the ML and DL models included, therefore the user does not need to merge gathered data anymore. PM25_MI_Marche_ARPA    contains the closest authoritative PM2.5 measurements for the webcam images that are used as inputs for the physical model.

- **processing outputs/**  
  Outcomes will be saved in this path.

- **01-visual_comparison/**  
  This notebook provides an interactive viewer to compare two webcam images side-by-side using a swipe overlay with multiple viewing modes (RGB, HSV, individual color channels, saturation, and blue/red ratio) and adjustable region-of-interest cropping.

- **02-image_feature_extraction/**  
  This notebook processes a folder of webcam images by cropping out overlay ribbons, splitting each image into sky and ground regions, and extracting mean RGB values for each region to export as a CSV file.

- **03-physical_based_model/**  
  This notebook implements a physical-based model that estimates PM2.5 concentrations from webcam sky images by extracting spatial and wavelet-transform entropy features from saturation channels, fitting reference distributions on clean-air samples, computing a combined quality score, and calibrating a logistic regression to predict PM2.5 levels.

- **04-pm25_rgb_analysis/**  
  This notebook analyzes and visualizes the relationship between PM2.5 concentrations and RGB channel values from sky and ground regions of webcam images.

- **05-decision_tree/**  
  This notebook trains a Decision Tree Regressor to predict PM2.5 concentrations from webcam RGB features, meteorological variables, and engineered energy terms, with separate day/night models.

- **06-mlp/**  
  This notebook builds and trains a Multi-Layer Perceptron neural network to predict PM2.5 concentrations using meteorological variables, atmospheric energy parameters, and webcam-derived RGB color features with cyclical time encoding.

## Notes

- Notebooks are documented inline to clarify assumptions and parameter choices.
- Methods are implemented using open-source Python libraries only.
- Results are meant for **methodological comparison**, not operational air quality monitoring.
