# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Define a function to handle encoding and missing value filling
def process_column(data, column_name, mapping=None, method=None):
    if mapping:
        encoded_column_name = f"{column_name}_encoded"
        data[encoded_column_name] = data[column_name].map(mapping).fillna(0).astype(int)
    if method:
        filled_column_name = f"{column_name}_filled"
        if method == "median":
            data[filled_column_name] = data[column_name].fillna(data[column_name].median())
        elif method == "mode":
            data[filled_column_name] = data[column_name].fillna(data[column_name].mode()[0])
    return data


# Load the dataset
file_path = './train.csv'
train_data = pd.read_csv(file_path)

print(len(train_data.columns))

# Define mappings and methods for columns
config = {
    # 'PoolQC': {'mapping': {'Ex': 3, 'Gd': 2, 'Fa': 1}, 'method': None},
    # 'Fence': {'mapping': {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1}, 'method': None},
    # 'MiscFeature': {'mapping': {'Shed': 1, 'Gar2': 2, 'Othr': 3, 'TenC': 4}, 'method': None},
    # 'Alley': {'mapping': {'Grvl': 1, 'Pave': 2}, 'method': None},
    # 'FireplaceQu': {'mapping': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}, 'method': None},
    # 'GarageType': {'mapping': {'Attchd': 1, 'Detchd': 2, 'BuiltIn': 3, 'CarPort': 4, 'Basment': 5, '2Types': 6}, 'method': None},
    # 'GarageFinish': {'mapping': {'Fin': 3, 'RFn': 2, 'Unf': 1}, 'method': None},
    # 'GarageQual': {'mapping': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}, 'method': None},
    # 'GarageCond': {'mapping': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}, 'method': None},
    # 'BsmtExposure': {'mapping': {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1}, 'method': None},
    # 'BsmtQual': {'mapping': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}, 'method': None},
    # 'BsmtCond': {'mapping': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}, 'method': None},
    # 'BsmtFinType1': {'mapping': {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1}, 'method': None},
    # 'MasVnrType': {'mapping': {'BrkFace': 3, 'None': 1, 'Stone': 4, 'BrkCmn': 2}, 'method': None},
    # 'Electrical': {'mapping': {'SBrkr': 5, 'FuseA': 4, 'FuseF': 3, 'FuseP': 2, 'Mix': 1}, 'method': None},
    'LotFrontage': {'mapping': None, 'method': "median"},
    'GarageYrBlt': {'mapping': None, 'method': "median"},
    'MasVnrArea': {'mapping': None, 'method': "mode"}
}



for column_name in train_data.columns:
	if column_name in list(config.keys()):
		continue
	elif 'encoded' in column_name:
		continue
	elif column_name == 'SalePrice':
		continue
	else:
		data = train_data[column_name].value_counts()
		if data.index.dtype != 'int64' and df.index.dtype != 'float64':
			# 假設你有以下的元組
			tuples = list(enumerate(data.index, start=1))

			# 使用 dict() 將元組轉換為字典，將文字作為 key，數字作為 value
			mapping = {'mapping': dict((value, key) for key, value in tuples), 'method': None }
			config[data.index.name] = mapping
			# a.plot(kind='bar')
			# plt.show()



# Apply processing to all columns in config
for column, settings in config.items():
    train_data = process_column(train_data, column, settings['mapping'], settings['method'])

# Drop original columns that have been processed
columns_to_drop = list(config.keys())
train_data.drop(columns=columns_to_drop, inplace=True)

print(len(train_data.columns))

# Save the processed data
output_path = './processed_train_data.csv'
train_data.to_csv(output_path, index=False)




