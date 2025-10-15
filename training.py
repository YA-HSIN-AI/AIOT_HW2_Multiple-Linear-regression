import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


y_test = [3, -0.5, 2, 7]
predictions = [2.5, 0.0, 2, 8]

try:
    rmse = mean_squared_error(y_test, predictions)
    print(f"RMSE: {rmse}")
except TypeError as e:
    print(f"Error: {e}")


import xgboost as xgb

def mean_absolute_error_percentage(y_true, y_pred):
    """
    Calculate Mean Absolute Error Percentage (MAEP).

    Parameters:
    y_true (array-like): True values
    y_pred (array-like): Predicted values

    Returns:
    float: MAEP in percentage
    """
    return (abs(y_pred - y_true) / y_true).mean() * 100


# 讀取數據集 (將 'your_dataset.csv' 替換為實際的檔案路徑)
data = pd.read_csv(r'C:\Users\User\Downloads\file\file\processed_train_data.csv')
#data = pd.read_csv('processed_train_data.csv')

# 假設 'target' 是目標變數，其餘為特徵
X = data.drop(columns=['SalePrice', 'Id'])
# X = data[['YearBuilt',   'YearRemodAdd',    
#                         # 'BsmtFinSF1', 
#                         # 'TotalBsmtSF',  
#                         '1stFlrSF',    '2ndFlrSF',    
#                         'GrLivArea',   #'GarageArea',  
#                         'MSSubClass_encoded',
#                         'LandContour_encoded', 'LandSlope_encoded',   
#                         'BldgType_encoded',    'HouseStyle_encoded',  
#                         'OverallCond_encoded', 'RoofStyle_encoded',   
#                         'RoofMatl_encoded',    'Exterior1st_encoded',
#                         'Exterior2nd_encoded', 'ExterQual_encoded',   
#                         'Foundation_encoded',  'HeatingQC_encoded',   
#                         'HalfBath_encoded',    'KitchenQual_encoded', 
#                         'GarageCars_encoded']]

y = data['SalePrice']

# 將數據分成訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 轉換成 XGBoost DMatrix 格式
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 設定 XGBoost 模型參數
params = {
    'objective': 'reg:squarederror',  # 使用均方誤差作為目標函數
    'eval_metric': 'rmse',           # 使用 RMSE 作為評估指標
    'max_depth': 5,                  # 樹的最大深度
    'eta': 0.1,                      # 學習率
    'subsample': 0.8,                # 每棵樹的樣本比例
    'colsample_bytree': 0.6         # 每棵樹的特徵比例
}

#Best Parameters: {'objective': 'reg:squarederror', 'max_depth': 5, 'eta': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.6}
#Best RMSE: 26856.35626617825

# 訓練模型
evals = [(dtrain, 'train'), (dtest, 'eval')]
model = xgb.train(params, dtrain, num_boost_round=100, evals=evals, early_stopping_rounds=10, verbose_eval=True)



# 預測
predictions = model.predict(dtest)

print(model.get_score())

# 評估模型

rmse = mean_squared_error(y_test, predictions)
print(f"RMSE: {rmse}")
mae = mean_absolute_error(y_test, predictions)
print(f"mae: {mae}")
maep = mean_absolute_error_percentage(y_test, predictions)
print(f"Mean Absolute Error Percentage (MAEP): {maep:.2f}%")

# 保存模型
model.save_model('xgboost_model.json')


# Load the test dataset
test_file_path = r'C:\Users\User\Downloads\file\file\test_processed_train_data.csv'

test_data = pd.read_csv(test_file_path)

test_data_11 = test_data[['YearBuilt',   'YearRemodAdd',    
                        # 'BsmtFinSF1', 
                        # 'TotalBsmtSF',  
                        '1stFlrSF',    '2ndFlrSF',    
                        'GrLivArea',   #'GarageArea',  
                        'MSSubClass_encoded',
                        'LandContour_encoded', 'LandSlope_encoded',   
                        'BldgType_encoded',    'HouseStyle_encoded',  
                        'OverallCond_encoded', 'RoofStyle_encoded',   
                        'RoofMatl_encoded',    'Exterior1st_encoded',
                        'Exterior2nd_encoded', 'ExterQual_encoded',   
                        'Foundation_encoded',  'HeatingQC_encoded',   
                        'HalfBath_encoded',    'KitchenQual_encoded', 
                        'GarageCars_encoded']]

test_data_11 = test_data.drop(columns=['Id'])
# print(X.columns)
# print(test_data_11.columns)


# Prepare the test dataset for prediction
# Assuming that the target variable is not included in the test dataset
dtest_new = xgb.DMatrix(test_data_11)

# Predict using the trained model
predictions_new = model.predict(dtest_new)

# Save the predictions to a CSV file
output = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': predictions_new})
output_file_path = './predictions.csv'
output.to_csv(output_file_path, index=False)

output_file_path


'''
RMSE: 692663104.0
mae: 16462.140625
Mean Absolute Error Percentage (MAEP): 10.06%
luyaoji@LuYaoJideMacBook-Air 2024_12_22_house_price % 

RMSE: 720845696.0
mae: 16512.91796875
Mean Absolute Error Percentage (MAEP): 9.86%
'''
