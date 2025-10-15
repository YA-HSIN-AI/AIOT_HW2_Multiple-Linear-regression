# 🏡 House Prices: Advanced Regression Techniques
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-green.svg)](https://xgboost.readthedocs.io/)
[![Kaggle](https://img.shields.io/badge/Kaggle-House%20Prices-blue.svg)](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## 📘 Project Description
A complete **machine learning pipeline** for predicting house sale prices using **XGBoost** and the **CRISP-DM** methodology.  
Based on the [Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) dataset, this project demonstrates data preprocessing, feature encoding, model training, evaluation, and Kaggle submission generation.

---

## 🧭 CRISP-DM Workflow

### 1️⃣ Business Understanding
- **Goal:** Predict the `SalePrice` of residential homes.  
- **Value:** Enables data-driven pricing decisions for real estate stakeholders.  
- **Metric:** Root Mean Squared Error (RMSE), aligning with Kaggle’s evaluation metric.

---

### 2️⃣ Data Understanding
**Dataset:** `train.csv` from Kaggle.  
Includes **80 features** and **1 target (`SalePrice`)** describing:
- Structural attributes (`YearBuilt`, `GrLivArea`, `GarageArea`)
- Location features (`Neighborhood`, `LotFrontage`)
- Quality assessments (`ExterQual`, `BsmtQual`, `KitchenQual`)

Common missing values:
- `LotFrontage`
- `GarageYrBlt`
- `MasVnrArea`

Exploration involved counting null values and identifying categorical/ordinal features that required mapping.

---

### 3️⃣ Data Preparation (`preprocessing.py`)
Data preprocessing follows structured feature engineering:

| Step | Operation | Example Columns | Method |
|------|------------|-----------------|--------|
| Missing Value Imputation | Fill missing numerical data | `LotFrontage`, `GarageYrBlt` | Median |
| | Fill missing categorical data | `MasVnrArea` | Mode |
| Ordinal Encoding | Convert categorical → numerical | `ExterQual`, `BsmtQual`, etc. | Manual mapping |
| Cleanup | Drop raw text columns | All mapped features | — |

✅ **Output:** `processed_train_data.csv`

**Run:**
```bash
python preprocessing.py


