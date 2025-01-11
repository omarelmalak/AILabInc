import pandas as pd
import os
import numpy as np


def load_student_metadata(root_dir="./data", num_users=None):
    """
    Load and preprocess student metadata.

    :param root_dir: str, root directory of the data.
    :param num_users: int, number of users to align metadata with.
    :return: Numpy array of processed student metadata.
    """
    path = os.path.join(root_dir, "student_meta.csv")
    if not os.path.exists(path):
        raise Exception(f"The specified path {path} does not exist.")

    student_meta = pd.read_csv(path)
    student_meta = student_meta.drop(columns=['data_of_birth'])

    student_meta['gender'] = student_meta['gender'].map({0: np.nan,1: 0,2: 1})
    if num_users:
        student_meta = student_meta.sort_values(by='user_id').set_index('user_id')
        student_meta = student_meta.reindex(range(num_users), fill_value=0)

    return student_meta.astype(float).values




def align_metadata_with_sparse_matrix(matrix, student_meta=None):
    """
    Align metadata with the given sparse matrix for augmentation.

    :param matrix: 2D numpy array (sparse matrix of user-question interactions).
    :param student_meta: Processed student metadata (rows of the matrix).
    :param question_meta: Processed question metadata (columns of the matrix).
    :return: Augmented matrix.
    """
    if student_meta is not None:
        matrix = np.hstack([matrix, student_meta])
    return matrix
