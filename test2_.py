import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

import xgboost as xgb
# from xgboost import XGBRegressor

# 自定義 MAEP 計算函數
def mean_absolute_error_percentage(y_true, y_pred):
    return (abs(y_pred - y_true) / y_true).mean() * 100

# 讀取數據集
data = pd.read_csv(r'C:\Users\User\Downloads\file\file\processed_train_data.csv')
#data = pd.read_csv('C:\Users\User\Downloads\file\file\processed_train_data.csv')
X = data.drop(columns=['SalePrice', 'Id'])
y = data['SalePrice']

# 分割訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 定義參數範圍
param_grid = {
    'max_depth': [3, 4, 5, 6, 7, 9, 10],

    'eta': [0.01, 0.05, 0.1],
    'n_estimators': [500, 1000, 1500],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],

    'learning_rate': [0.01, 0.03, 0.1],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 10, 100]
}

best_params = None
best_rmse = float('inf')

# 遍歷參數
for max_depth in param_grid['max_depth']:
    for eta in param_grid['eta']:
        for subsample in param_grid['subsample']:
            for colsample_bytree in param_grid['colsample_bytree']:
                params = {
                    'objective': 'reg:squarederror',
                    'max_depth': max_depth,
                    'eta': eta,
                    'subsample': subsample,
                    'colsample_bytree': colsample_bytree
                }
                cv_results = xgb.cv(
                    params=params,
                    dtrain=dtrain,
                    num_boost_round=100,
                    nfold=5,
                    metrics='rmse',
                    early_stopping_rounds=10,
                    seed=42
                )
                mean_rmse = cv_results['test-rmse-mean'].min()
                if mean_rmse < best_rmse:
                    best_rmse = mean_rmse
                    best_params = params

print("Best Parameters:", best_params)
print("Best RMSE:", best_rmse)
