import matplotlib.pyplot as plt
import numpy as np

def analyze_and_visualize_feature_correlation(original_data):
    correlation_matrix = original_data.corr()

    # 可视化相关性矩阵
    plt.figure(figsize=(12, 10))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title('特征相关性矩阵')
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.show()

    # 找出高度相关的特征对
    highly_correlated = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > 0.7:
                highly_correlated.append((correlation_matrix.columns[i],
                                          correlation_matrix.columns[j],
                                          correlation_matrix.iloc[i, j]))

    print("\n高度相关的特征对（相关系数 > 0.7）:")
    for pair in highly_correlated:
        print(f"{pair[0]} & {pair[1]}: {pair[2]:.2f}")