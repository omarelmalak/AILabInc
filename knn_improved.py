import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.decomposition import PCA
import time

from sklearn.manifold import TSNE

from utils import (
    load_train_sparse,
    load_valid_csv,
    load_public_test_csv,
    sparse_matrix_evaluate,
)


def tsne_visualize(matrix, n_vals):
    preprocessed_matrix = process_data(matrix)

    tsne = TSNE(n_components=2, random_state=42)
    tsne_transformed = tsne.fit_transform(preprocessed_matrix)

    plt.figure(figsize=(10, 5))
    plt.scatter(tsne_transformed[:, 0], tsne_transformed[:, 1], alpha=0.5)
    plt.title("t-SNE Visualization of Original Data")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.savefig("tsne_original.png", dpi=300)
    plt.show()

    for n in n_vals:
        _, pca = apply_pca(preprocessed_matrix, n)
        pca_transformed = pca.transform(preprocessed_matrix)
        tsne_after_pca = tsne.fit_transform(pca_transformed)

        plt.figure(figsize=(10, 5))
        plt.scatter(tsne_after_pca[:, 0], tsne_after_pca[:, 1], alpha=0.5, color='orange')
        plt.title(f"t-SNE Visualization After PCA (n_components={n})")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True)
        plt.savefig(f"tsne_after_pca_{n}.png", dpi=300)
        plt.show()


def process_data(matrix):
    imputer = SimpleImputer(strategy='mean')
    imputed_matrix = imputer.fit_transform(matrix)
    return imputed_matrix


def apply_pca(matrix, n_components):
    pca = PCA(n_components=n_components)
    pca_transformed = pca.fit_transform(matrix)
    return pca_transformed, pca


def knn_pca(matrix, valid_data, k, n_components):
    matrix_T = matrix.T
    imputed_mat_T = process_data(matrix_T)

    pca_transformed, pca = apply_pca(imputed_mat_T, n_components)

    knn_imputer = KNNImputer(n_neighbors=k)
    imputed_pca = knn_imputer.fit_transform(pca_transformed)

    full_imputed_T = pca.inverse_transform(imputed_pca)

    imputed_full = full_imputed_T.T

    acc = sparse_matrix_evaluate(valid_data, imputed_full)
    print(f"Validation Accuracy with k={k}, n_components={n_components}: {acc}")
    return acc


def main():
    start_time = time.time()
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("KNN with PCA:")
    k_values = [1, 6, 11, 16, 21, 26]
    n_components = [5, 10, 20, 50, 100, 200, 300, 400, 500, 542]
    best_accuracy = 0
    best_k = None
    best_n = None

    tsne_visualize(sparse_matrix, n_components)

    for n in n_components:
        accuracies = []
        for k in k_values:
            acc = knn_pca(sparse_matrix, val_data, k, n)
            accuracies.append(acc)
            if acc > best_accuracy:
                best_accuracy = acc
                best_k = k
                best_n = n

        plt.plot(k_values, accuracies, marker="o", label=f"n_components={n}")

    plt.title("Validation Accuracy vs k (KNN with PCA)")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("validation_accuracy_vs_k.png", dpi=300, format='png')
    plt.show()

    print(f"best validation accuracy: {best_accuracy} with k={best_k} and n_components={best_n}")

    test_acc = knn_pca(sparse_matrix, test_data, best_k, best_n)
    print(f"test accuracy for KNN with k={best_k} and n_components={best_n}: {test_acc}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"duration: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
