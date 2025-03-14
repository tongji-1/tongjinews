from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def select_optimal_clusters(principal_components, max_clusters=10):
    # 使用肘部法则和轮廓系数选择最优聚类数量
    inertias = []
    silhouette_scores = []

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(principal_components)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(principal_components, kmeans.labels_))

    # 找到肘部点和最高轮廓系数对应的聚类数量
    diff = np.diff(inertias)
    acceleration = np.diff(diff)
    optimal_k_elbow = np.argmax(acceleration) + 2

    optimal_k_silhouette = np.argmax(silhouette_scores) + 2

    optimal_k = (optimal_k_elbow + optimal_k_silhouette) // 2
    optimal_k = max(2, min(optimal_k, max_clusters))

    print(f"\n肘部法则建议聚类数量: {optimal_k_elbow}")
    print(f"轮廓系数建议聚类数量: {optimal_k_silhouette}")
    print(f"综合建议聚类数量: {optimal_k}")

    # 可视化肘部法则和轮廓系数
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_clusters + 1), inertias, marker='o')
    plt.xlabel('聚类数量')
    plt.ylabel('惯性')
    plt.title('肘部法则')

    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.xlabel('聚类数量')
    plt.ylabel('轮廓系数')
    plt.title('轮廓系数')

    plt.tight_layout()
    plt.show()

    return optimal_k

def perform_clustering(principal_components, optimal_k):
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans.fit_predict(principal_components)
    return cluster_labels