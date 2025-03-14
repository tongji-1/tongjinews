import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visualize_pca_3d(principal_components, cluster_labels, n_components):
    if n_components >= 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for i in range(len(set(cluster_labels))):
            ax.scatter(principal_components[cluster_labels == i, 0],
                       principal_components[cluster_labels == i, 1],
                       principal_components[cluster_labels == i, 2],
                       label=f'Cluster {i + 1}', alpha=0.7)

        ax.set_xlabel('主成分1')
        ax.set_ylabel('主成分2')
        ax.set_zlabel('主成分3')
        ax.set_title('3D PCA与聚类结果')
        ax.legend()
        plt.show()
    else:
        plt.figure(figsize=(10, 6))
        for i in range(len(set(cluster_labels))):
            plt.scatter(principal_components[cluster_labels == i, 0],
                        principal_components[cluster_labels == i, 1],
                        label=f'Cluster {i + 1}', alpha=0.7)

        plt.xlabel('主成分1')
        plt.ylabel('主成分2')
        plt.title('2D PCA与聚类结果')
        plt.legend()
        plt.show()