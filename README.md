Computer Vision–Based Multiclass Classification Using Morphological Features, Naive Bayes, and PCA (MATLAB)
Overview

This project implements a complete computer vision and statistical learning pipeline for multiclass object classification using RGB images. The workflow includes:

Image segmentation

Morphological and color feature extraction

Statistical hypothesis testing

Gaussian Naive Bayes classification

Dimensionality reduction using Principal Component Analysis (PCA)

The goal was to evaluate the discriminative power of handcrafted morphological features and compare classification performance before and after dimensionality reduction.

Dataset

16 RGB images

8 object classes

25 objects per image

Total: 400 segmented objects

Objects were imaged against a uniform background to facilitate segmentation.

Methodology
1. Segmentation Strategies

Two segmentation approaches were implemented and compared:

Method A — Automated Segmentation (Recommended)

RGB normalization (normr)

Color index transformation: gray = r + g − b

Otsu thresholding (graythresh)

Morphological refinement (bwmorph)

Connected-component labeling (bwlabel)

This approach is fully reproducible and requires no manual interaction.

Method B — ROI-Based Mahalanobis Segmentation

Manual background sampling (roipoly)

RGB covariance estimation

Mahalanobis distance–based segmentation

Morphological refinement

This method demonstrates color-space statistical segmentation but is semi-supervised.

Feature Extraction

For each object, the following features were computed:

Morphological Features

Area

Perimeter

MajorAxisLength

MinorAxisLength

Eccentricity

Solidity

Elongation

AspectRatio

Compactness

Roundness

Color Features

Hue (HSI color space)

Intensity

Total features per object: 12

Classification

70% training / 30% testing split

Gaussian Naive Bayes classifier (fitcnb)

Confusion matrix analysis

Principal Component Analysis (PCA)

PCA applied to reduce dimensionality

First 3 principal components selected (>90% variance explained)

Classification repeated using PCA-reduced features

Results

Naive Bayes with full feature set achieved higher classification accuracy.

Accuracy decreased after PCA dimensionality reduction.

This suggests that certain handcrafted morphological features carry highly discriminative class-specific information that may be partially lost under linear projection.

Key Insights

Feature engineering remains powerful for structured image classification problems.

Dimensionality reduction does not always improve classification performance.

Morphological descriptors can effectively discriminate visually similar classes.

Technologies

MATLAB

Image Processing Toolbox

Statistics and Machine Learning Toolbox

Potential Extensions

k-fold cross-validation

Comparison with SVM and Random Forest

Adaptation to biomedical image analysis (e.g., cell morphology, lesion detection)

Integration with deep learning feature extractors
