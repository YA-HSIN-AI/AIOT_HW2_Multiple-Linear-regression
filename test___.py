import xgboost as xgb
from sklearn.model_selection import train_test_split

# 數據準備
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 定義參數範圍
param_grid = {
    'max_depth': [3, 5, 7],
    'eta': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
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
