import numpy as np
from sklearn.impute import KNNImputer
import torch
from torch.autograd import Variable

from item_response import sigmoid, irt
from utils import load_train_sparse, load_valid_csv, load_public_test_csv, load_train_csv, sparse_matrix_evaluate
from neural_network import load_data, AutoEncoder, train
from sklearn.utils import resample


def knn_predict_by_user(matrix, k):
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix)
    return mat


def irt_predict(data, theta, beta):
    num_users = max(data['user_id']) + 1
    num_questions = max(data['question_id']) + 1

    matrix = np.full((num_users, num_questions), np.nan)

    for user_id, question_id in zip(data['user_id'], data['question_id']):
        matrix[user_id, question_id] = sigmoid(theta[user_id] - beta[question_id])
    matrix = np.nan_to_num(matrix, nan=0.5)

    return matrix


def nn_predict(model, data):
    model.eval()
    predictions = []

    with torch.no_grad():
        for user_id in range(data.shape[0]):
            inputs = Variable(data[user_id]).unsqueeze(0)
            output = model(inputs)
            predictions.append(output.squeeze().numpy())

    return np.array(predictions)


def bootstrap_data(data):
    n_samples = len(data['user_id'])
    indices = np.random.choice(n_samples, n_samples, replace=True)

    bootstrapped_data = {
        'user_id': [],
        'question_id': [],
        'is_correct': []
    }

    for i in indices:
        bootstrapped_data['user_id'].append(data['user_id'][i])
        bootstrapped_data['question_id'].append(data['question_id'][i])
        bootstrapped_data['is_correct'].append(data['is_correct'][i])

    return bootstrapped_data


def bootstrap_nn_data(train_matrix):
    indices = resample(train_matrix, replace=True)
    return indices


def bootstrap_knn_data(sparse_matrix):
    n_samples = sparse_matrix.shape[0]
    indices = np.random.choice(n_samples, n_samples, replace=True)
    return sparse_matrix[indices]


# KNN
sparse_matrix = load_train_sparse("./data").toarray()
bootstrapped_matrix = bootstrap_knn_data(sparse_matrix)
k = 11
predicted_matrix = knn_predict_by_user(bootstrapped_matrix, k)


# IRT
train_data = load_train_csv("./data")
bootstrapped_train_data = bootstrap_data(train_data)
lr = 0.001
iterations = 100
theta, beta, _, _, _ = irt(bootstrapped_train_data, train_data, lr, iterations)
irt_predictions = irt_predict(train_data, theta, beta)
irt_matrix = irt_predictions



# NN
zero_train_matrix, train_matrix, valid_data, test_data = load_data()
bootstrapped_train_matrix = bootstrap_nn_data(train_matrix)
model = AutoEncoder(len(train_matrix[0]), k)

k = 200
nn_lr = 0.01
num_epoch = 10
lamb = 0.001

train(model, nn_lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)

nn_predictions = nn_predict(model, zero_train_matrix)



# ensemble part
average_predictions = (predicted_matrix + irt_matrix + nn_predictions) / 3
average_individual_variance = np.var(average_predictions, axis=0).mean()
# print(f"Variance (After Bagging): {average_individual_variance:.4f}")
#
#
# print("averaged predictions:")
# print(average_predictions[:10])
#
# print("avg predictions shape:", average_predictions.shape)
# print(f"min val: {average_predictions.min()}, max val: {average_predictions.max()}")
#
print("Validation accuracy (KNN):", sparse_matrix_evaluate(valid_data, predicted_matrix))
print("Validation accuracy (IRT):", sparse_matrix_evaluate(valid_data, irt_predictions))
print("Validation accuracy (NN):", sparse_matrix_evaluate(valid_data, nn_predictions))

print("Validation accuracy: ", sparse_matrix_evaluate(valid_data, average_predictions))
print("Test Accuracy: ", sparse_matrix_evaluate(test_data, average_predictions))
