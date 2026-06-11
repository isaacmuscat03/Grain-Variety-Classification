# Classification of Grain Varieties Using Hyperspectral Imaging and Deep Learning

This repository contains the code developed as part of the Master's thesis on grain variety classification using hyperspectral imaging (HSI).

The aim of the project is to investigate how much spectral and spatial information is required to reliably distinguish between different grain varieties, and how these findings can inform the design of reduced-band multispectral imaging systems.

## Project Structure

### `data_preparation.ipynb`

Data loading and preprocessing pipeline:

* Loading and calibration of hyperspectral cubes
* Foreground masking and background removal
* Spatial cropping
* Pixel sampling
* Dataset creation and train/validation/test split

### `spectral_exploration.ipynb`

Exploratory spectral analysis:

* Mean spectra visualisation
* Spectral preprocessing and mean centering
* Principal Component Analysis (PCA)
* PCA score images
* PCA loading analysis
* Investigation of informative wavelengths and spectral regions

### `spectral_classification.ipynb`

Pixel-level spectral classification experiments:

* PLS-DA baseline models
* Reduced-band classification using PCA-selected wavelengths
* Model comparison across different spectral subsets
* Confusion matrices and performance metrics
* Diagnostic analyses of class-specific behaviour and misclassifications

### `spatial_classification.ipynb`

Spatial modelling experiments:

* Conversion of hyperspectral cubes to image representations
* Convolutional Neural Networks (CNNs)
* ResNet-based image classification
* Investigation of the contribution of spatial information under extreme spectral reduction (1 band currently, 3 bands in the future)

### `hsi_utils.py`

Shared helper functions used throughout the project:

* Data loading
* Cube processing
* Foreground masking

