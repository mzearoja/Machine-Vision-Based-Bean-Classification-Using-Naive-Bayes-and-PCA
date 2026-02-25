# Machine-Vision-Based-Bean-Classification-Using-Naive-Bayes-and-PCA

Computer Vision–Based Multiclass Object Classification Using Morphological Features and Naive Bayes (MATLAB)

## Overview

This project implements a complete computer vision pipeline for multiclass classification of objects (bean varieties) using RGB images. The workflow includes image segmentation, morphological feature extraction, statistical hypothesis testing, Bayesian classification, and dimensionality reduction via Principal Component Analysis (PCA).

The objective was to evaluate how well morphological and color-based features discriminate between visually similar classes and to compare classification performance before and after dimensionality reduction.

---

## Dataset

- 16 RGB images
- 8 object classes
- 25 objects per image
- Total: 400 segmented objects

Each image contains objects placed on a uniform background to facilitate segmentation.

---

## Methodology

### 1. Image Segmentation

Two segmentation approaches were explored:

- Mahalanobis distance-based color segmentation
- Automatic thresholding (Otsu method) after RGB normalization

Morphological operations (erosion, dilation, majority filtering) were applied to refine segmentation masks.

---

### 2. Feature Extraction

For each segmented object, the following features were extracted:

**Morphological Features**
- Area
- Perimeter
- Major Axis Length
- Minor Axis Length
- Eccentricity
- Solidity
- Elongation
- Aspect Ratio
- Compactness
- Roundness

**Color Features**
- Hue (HSI color space)
- Intensity

Total: 12 features per object

---

### 3. Statistical Analysis

- Two-sample t-tests were performed to evaluate discriminative power of individual features.
- Pairwise comparisons across classes were conducted.

---

### 4. Classification – Naive Bayes

- 70% of samples used for training
- 30% used for testing
- Gaussian Naive Bayes classifier implemented using `fitcnb`

Confusion matrices were generated for performance evaluation.

---

### 5. Principal Component Analysis (PCA)

- PCA applied to reduce dimensionality
- First 3 principal components selected (>90% variance explained)
- Classification repeated using PCA-transformed features

---

## Results

- Naive Bayes using all 12 features achieved near-perfect classification.
- Performance decreased after PCA dimensionality reduction.
- This suggests that certain individual morphological features carry class-specific discriminative information that may be partially lost through linear projection.

---

## Key Insights

- Careful feature engineering can outperform dimensionality reduction in visually similar object classes.
- Morphological descriptors are highly informative for object classification.
- PCA is not always beneficial when fine-grained distinctions are required.

---

## Technologies Used

- MATLAB
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox

---

## Future Improvements

- Implement cross-validation
- Compare with SVM and Random Forest
- Automate background detection
- Extend to hyperspectral data
