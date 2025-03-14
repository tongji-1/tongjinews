from sklearn.decomposition import PCA
import numpy as np

def select_optimal_components(scaled_data, variance_threshold=0.95):
    # 计算累计解释方差比例
    pca = PCA()
    pca.fit(scaled_data)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # 找到满足方差阈值的最小主成分数量
    n_components = np.where(cumulative_variance >= variance_threshold)[0][0] + 1
    print(f"\n选择{variance_threshold * 100}%方差阈值，需要{n_components}个主成分")

    # 返回选择的主成分数量和对应的PCA模型
    pca_optimal = PCA(n_components=n_components)
    principal_components = pca_optimal.fit_transform(scaled_data)

    return principal_components, pca_optimal, n_components