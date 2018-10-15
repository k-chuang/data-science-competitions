# Traffic Image Classification
A data mining project that develops a predictive model that can determine, given images of traffic depicting different objects, which class (out of 11 total classes) it belongs to. The object classes are: car, suv, small_truck, medium_truck, large_truck, pedestrian, bus, van, people, bicycle, and motorcycle.

## Exploratory Data Analysis (EDA)
- Features already extracted from images
- Features include:
  - **512 Histogram of Oriented Gradients (HOG) features**
      - HOG counts occurrences of gradient orientation in localized portions of an image
  - **256 Normalized Color Histogram (Hist) features**
      - Histogram gives intensity of distribution of an image
      - Get intuition about contrast, brightness, intensity distribution etc of that image
  - **64 Local Binary Pattern (LBP) features**
      - LBP looks at points surrounding a central point and tests whether the surrounding points are greater than or less than the central point (i.e. gives a binary result).
      - Used for classifying textures (edges, corners, etc.)
  - **48 Color gradient (RGB) features**
      - Color gradient measures gradual change/blend of color within an image
  - **7 Depth of Field (DF) features**
      - Depth of Field is the distance about the plane of focus (POF) where objects appear acceptably sharp in an image

- Classes are imbalanced
  - 0 instances of human images
  - 3 instances of bicycle images
  - 10,375 instances of car images

## Feature Selection

- PCA (Unsupervised feature extraction)
    - Identify the combination of attributes (principal components) that account for the most variance in the data.
    - 95% kept variance = 14 components
    - 99% kept variance = 22 components
    - 100% kept variance = 34 components
- LDA (Supervised feature extraction)
    - Identify attributes that account for the most variance between classes
    - 100% kept variance = 9 components
- Locally Linear Embedding (Non-linear dimensionality reduction)
    - Uses PCA so using 34 components
- Removing Features with low variance `VarianceThreshold`
    - Throw away features with 0 variance
        - remove the features that have the same value in all samples.
- Tree-based feature selection
    - Use a meta-estimator to determine/learn feature importances, and use for feature selection
- L1-based feature selection (Lasso Regularization)
    - Linear models penalized with the L1 norm have sparse solutions: many of their estimated coefficients are zero.
    - For SVMs, parameter C controls the sparsity: the smaller C the fewer features selected.
- More to be added...

## Model Selection
- Use 5-Fold CV to determine performance of each model
- Models:
  - Support Vector Machine (Gaussian RBF kernel)
  - K-nearest neighbors
  - Ensemble Methods:
    - Random Forest
    - Light Gradient Boosting
    - Voting Classifier
