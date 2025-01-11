from matplotlib import pyplot as plt

from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np


def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    log_lklihood = 0.0

    for user_id, question_id, is_correct in zip(data['user_id'], data['question_id'], data['is_correct']):
        # This will find the probability of answering correctly
        theta_i = theta[user_id]
        beta_j = beta[question_id]
        p_correct = sigmoid(theta_i - beta_j)

        # Update the log-likelihood based on correctness
        if is_correct:
            log_lklihood += np.log(p_correct)
        else:
            log_lklihood += np.log(1 - p_correct)

    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    theta_grad = np.zeros_like(theta)
    beta_grad = np.zeros_like(beta)

    for user_id, question_id, is_correct in zip(data['user_id'], data['question_id'], data['is_correct']):
        # This will find the probability of answering correctly
        theta_i = theta[user_id]
        beta_j = beta[question_id]
        p_correct = sigmoid(theta_i - beta_j)

        # Compute the gradient contributions
        theta_grad[user_id] += (is_correct - p_correct)
        beta_grad[question_id] += (p_correct - is_correct)

    theta += lr * theta_grad
    beta += lr * beta_grad

    return theta, beta


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # Initialize theta and beta
    theta = np.random.normal(0, 1, max(data['user_id']) + 1)  # Since question 1's id is 0
    beta = np.random.normal(0, 1, max(data['question_id']) + 1)

    val_acc_list = []
    train_nllk_list = []
    val_nllk_list = []

    for i in range(iterations):
        # Compute negative log-likelihoods
        train_nllk = neg_log_likelihood(data, theta=theta, beta=beta)
        val_nllk = neg_log_likelihood(val_data, theta=theta, beta=beta)

        # Finds the validation accuracy
        val_acc = evaluate(val_data, theta, beta)
        val_acc_list.append(val_acc)

        # Record metrics for later plotting
        train_nllk_list.append(train_nllk)
        val_nllk_list.append(val_nllk)

        print("NLLK: {} \t Score: {}".format(train_nllk, val_acc))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    return theta, beta, val_acc_list, train_nllk_list, val_nllk_list


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    # Manually changing these hyperparameters and looking at resulting training curve
    lr = 0.001  # Tuned learning rate
    iterations = 300  # Tuned number of iterations

    # Train the model
    theta, beta, val_acc_list, train_nllk_list, val_nllk_list = irt(train_data, val_data, lr, iterations)

    # Report training curve
    plt.figure()
    plt.plot(range(iterations), train_nllk_list, label="Training")
    plt.plot(range(iterations), val_nllk_list, label="Validation", linestyle="--")
    plt.xlabel("Iterations")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("Training and Validation Log-Likelihoods as a Function of Iterations")
    plt.legend()
    plt.grid()
    plt.show()

    # Report final validation and test accuracy
    best_val_acc = val_acc_list[-1]
    test_acc = evaluate(test_data, theta, beta)
    print(f"Final Validation Accuracy: {best_val_acc}")
    print(f"Final Test Accuracy: {test_acc}")

    # Select three questions and plot probability curves
    selected_questions = [0, 133, 200]
    question_labels = ["j1", "j2", "j3"]
    theta_range = np.linspace(-3, 3, 100)

    plt.figure()
    for q, label in zip(selected_questions, question_labels):
        p = sigmoid(theta_range - beta[q])
        plt.plot(theta_range, p, label=f"Question {label} (β={beta[q]:.2f})")

    plt.xlabel("Student Competence (θ)")
    plt.ylabel("Probability of Correct Response")
    plt.title("Probability of Correct Response as a Function Student Competence for Three Questions")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
