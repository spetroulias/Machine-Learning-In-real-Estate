# Real Estate Investment Intelligence: Property Tier Classification & Price Prediction


## Executive Summary
This project delivers a complete machine learning system designed to optimize real estate investment decisions. By utilizing the Ames Housing dataset, the project develops two distinct models:
1. **A Classification Engine:** Identifies premium properties (top 25% of the market) to prevent missed acquisition opportunities.
2. **A Regression Engine:** Predicts exact sale prices to reduce pricing uncertainty and improve capital allocation.

By automating property tiering and valuation, this tool replaces manual appraisal heuristics, significantly reducing risk and increasing operational efficiency for property investors and asset managers.

---

## Dataset Overview
The project utilizes the **Ames Housing Dataset** (from the Kaggle House Prices competition), which contains public property sales records from Ames, Iowa.
* **Size:** 1,460 residential properties.
* **Features:** 81 raw columns detailing structural characteristics, quality ratings, lot dimensions, and neighborhood data.
* **Classification Target (`PriceCategory`):** Binary split into Standard (≤ 75th percentile) and Premium (> 75th percentile).
* **Regression Target (`SalePrice`):** Continuous property value in USD.

---

## Data Preprocessing & Feature Engineering
To ensure robust model performance and prevent data leakage, a strict preprocessing pipeline was implemented:
* **Feature Engineering:** Created a `HouseAge` feature (`YrSold` − `YearBuilt`) to directly capture property depreciation.
* **Imputation:** Applied median imputation for numerical features (to resist outliers) and most-frequent imputation for categorical features. High-missing-value columns (over 80% nulls) like PoolQC and Alley were dropped.
* **Outlier Handling:** Utilized IQR-based Winsorisation to cap extreme values in continuous features like `GrLivArea` and `TotalBsmtSF` without losing minority class rows.
* **Dimensionality Reduction:** Removed highly collinear features (>0.90 correlation) and applied Principal Component Analysis (PCA) to retain 95% of explained variance, preventing the curse of dimensionality.
* **Class Imbalance:** Addressed the 75/25 classification split by applying SMOTE exclusively to the training set, successfully generating synthetic Premium samples.

---

## Machine Learning Models & Results

### 1. Classification (Premium vs. Standard)
* **K-Nearest Neighbors (KNN):** Optimized via GridSearchCV, the application of SMOTE drastically improved the Premium class recall from 58% to 92%, significantly minimizing False Negatives (missed premium properties).
* **Random Forest Classifier:** Achieved the best overall performance with a 93% accuracy and a macro F1-Score of 0.91. The model successfully identified 83% of high-value properties while keeping False Positives to a minimum.

### 2. Regression (Price Prediction)
* **Multiple Linear Regression (OLS) & Lasso:** Lasso successfully performed automatic feature selection, reducing the feature space to 94 non-zero coefficients and achieving an R² of 0.897 with a Mean Absolute Error (MAE) of $15,758.
* **Random Forest Regressor:** Captured complex, non-linear market relationships to achieve the highest R² (0.900) and the lowest Root Mean Square Error (RMSE) at $22,890. 

---

## Explainable AI (XAI) - SHAP Analysis
To ensure institutional investment readiness, SHapley Additive exPlanations (SHAP) were integrated to provide transparency into model decision-making:
* **Top Classification Drivers:** `OverallQual` (material and finish quality) was identified as the single strongest predictor of a property achieving Premium tier status, followed by `GrLivArea` and `GarageCars`.
* **Top Regression Drivers:** `GrLivArea` (above-ground living area) exhibited the largest continuous impact on predicted dollar value.
* **Local Interpretability:** SHAP waterfall plots were generated to decompose individual property valuations into exact marginal feature contributions, explaining precisely why a specific home was priced or tiered the way it was.
