import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def knn_impute_by_user(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    matrix_transposed = matrix.T
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix_transposed)
    mat = mat.T
    acc = sparse_matrix_evaluate(valid_data, mat)

    print("Validation Accuracy: {}".format(acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    print("User-based KNN:")

    k_values = [1,6,11,16,21,26]
    val_accuracies = []

    for k in k_values:
        acc = knn_impute_by_user(sparse_matrix,val_data, k)
        val_accuracies.append(acc)

    k_star = k_values[np.argmax(val_accuracies)]
    print(f"Best k for user-based: {k_star}")

    test_acc = knn_impute_by_user(sparse_matrix, test_data, k_star)
    print(f"Test Accuracy for k={k_star}: {test_acc}")

    plt.plot(k_values, val_accuracies, marker="o")
    plt.title("Validation Accuracy vs k")
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("Validation Accuracy")
    plt.grid(True)
    plt.show()




    # ITEM_BASED
    print("\n\nItem-based KNN:")
    val_accuracies_item = []
    for k in k_values:
        acc = knn_impute_by_item(sparse_matrix, val_data, k)
        val_accuracies_item.append(acc)

    k_star_item = k_values[np.argmax(val_accuracies_item)]
    print(f"Best k for user-based: {k_star_item}")

    test_acc_item = knn_impute_by_item(sparse_matrix, test_data, k_star_item)
    print(f"Test Accuracy for k={k_star_item}: {test_acc_item}")

    plt.plot(k_values, val_accuracies_item, marker="o")
    plt.title("Item-based KNN Validation Accuracy vs k")
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("Validation Accuracy")
    plt.grid(True)
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
