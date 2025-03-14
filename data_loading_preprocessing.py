import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    # 读取Excel文件
    df = pd.read_excel(file_path, header=0, index_col=0)

    # 数据概览
    print("数据集形状:", df.shape)
    print("\n前5行数据:")
    print(df.head())

    # 数据标准化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    return df, scaled_data