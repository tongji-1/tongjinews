import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loading_preprocessing import load_and_preprocess_data
from pca_analysis import select_optimal_components
from clustering_analysis import select_optimal_clusters, perform_clustering
from visualization import visualize_pca_3d
from feature_correlation_analysis import analyze_and_visualize_feature_correlation

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

if __name__ == "__main__":
    # 替换为您的Excel文件路径
    file_path = 'summaries_gpt.xlsx'

    # 1. 数据加载与预处理
    original_data, scaled_data = load_and_preprocess_data(file_path)

    # 2. 自动选择主成分数量
    principal_components, pca_optimal, n_components = select_optimal_components(scaled_data)

    # 3. 自动选择聚类数量
    optimal_k = select_optimal_clusters(principal_components)

    # 4. 聚类分析
    cluster_labels = perform_clustering(principal_components, optimal_k)

    # 5. 可视化PCA与聚类结果
    visualize_pca_3d(principal_components, cluster_labels, n_components)

    # 6. 特征关联度分析与可视化
    analyze_and_visualize_feature_correlation(original_data)