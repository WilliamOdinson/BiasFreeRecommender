import numpy as np
from sklearn.metrics import mean_squared_error


def compute_approximation_error(num_singular_values, original_matrix, U, S, Vt):
    """
    计算使用前 num_singular_values 个奇异值重构矩阵后的近似误差。

    Args:
        num_singular_values (int): 使用的奇异值数量
        original_matrix (ndarray): 原始矩阵
        U (ndarray): 奇异值分解得到的左奇异矩阵
        S (ndarray): 奇异值分解得到的奇异值矩阵
        Vt (ndarray): 奇异值分解得到的右奇异矩阵的转置

    Returns:
        float: 近似误差的均方差
    """
    # 重构矩阵
    reconstructed_matrix = np.dot(U[:, :num_singular_values], np.dot(
        S[:num_singular_values, :num_singular_values], Vt[:num_singular_values, :]))
    # 获取原始矩阵中非零元素的索引
    non_zero_indices = np.where(original_matrix > 0)
    # 计算原始矩阵和重构矩阵之间的差异
    difference = original_matrix[non_zero_indices] - \
        reconstructed_matrix[non_zero_indices]
    # 返回差异的均方差
    return np.linalg.norm(difference)**2 / difference.shape[1]


def compute_global_user_item_bias(rating_matrix):
    """
    计算用户和物品的全局偏差，并返回去除偏差的评分矩阵。

    Args:
        rating_matrix (scipy.sparse matrix): 用户-物品评分矩阵

    Returns:
        tuple: 包含用户偏差、物品偏差和去除偏差的评分矩阵的元组
    """
    dense_matrix = rating_matrix.todense()
    mask_matrix = (dense_matrix > 0).astype(int)

    # 计算用户偏差
    user_bias = np.sum(dense_matrix, axis=1) / np.sum(mask_matrix, axis=1)
    user_bias = np.nan_to_num(user_bias).reshape(-1, 1)

    # 计算物品偏差，避免除以零
    item_bias_denominator = np.sum(mask_matrix, axis=0)
    item_bias_denominator[item_bias_denominator == 0] = 1  # 避免除以零
    item_bias = np.sum(dense_matrix, axis=0) / item_bias_denominator
    item_bias = np.nan_to_num(item_bias).reshape(1, -1)

    # 计算去除偏差的评分矩阵
    ratings_matrix_no_bias = dense_matrix - \
        np.tile(user_bias, (1, dense_matrix.shape[1])) - \
        np.tile(item_bias, (dense_matrix.shape[0], 1))

    return user_bias, item_bias, ratings_matrix_no_bias


def compute_similarity_matrix(rating_matrix, axis='users'):
    """
    计算用户或物品的余弦相似度矩阵。

    Args:
        rating_matrix (scipy.sparse matrix): 用户-物品评分矩阵
        axis (str): 'users' 表示计算用户之间的相似度，'items' 表示计算物品之间的相似度

    Returns:
        numpy.matrix: 余弦相似度矩阵
    """
    if axis == 'users':
        similarity_matrix = rating_matrix * rating_matrix.transpose()
    else:
        similarity_matrix = rating_matrix.transpose() * rating_matrix

    dense_similarity_matrix = similarity_matrix.todense() + 1e-8
    norm_array = np.sqrt(np.diag(dense_similarity_matrix))

    return dense_similarity_matrix / norm_array / norm_array.reshape(-1, 1)


def predict_top_k(rating_matrix, similarity_matrix, kind='user', k=40):
    """
    使用余弦相似度矩阵进行Top-K预测。

    Args:
        rating_matrix (numpy.ndarray): 用户-物品评分矩阵
        similarity_matrix (numpy.matrix): 余弦相似度矩阵
        kind (str): 'user' 表示基于用户的预测，'item' 表示基于物品的预测
        k (int): 选择Top-K相似的用户或物品进行预测

    Returns:
        numpy.ndarray: 预测的评分矩阵
    """
    predictions = np.zeros(rating_matrix.shape)

    if kind == 'user':
        user_bias = np.mean(rating_matrix, axis=1)
        rating_matrix_adjusted = (
            rating_matrix - np.tile(user_bias, (rating_matrix.shape[1], 1)).T).copy()

        for i in range(rating_matrix.shape[0]):
            top_k_users = np.argsort(similarity_matrix[:, i])[:-k-1:-1]
            predictions[i] = np.dot(similarity_matrix[i, top_k_users], rating_matrix_adjusted[top_k_users, :]
                                    ) / np.sum(np.abs(similarity_matrix[i, top_k_users]))

        predictions += np.tile(user_bias, (rating_matrix.shape[1], 1)).T

    else:
        item_bias = np.mean(rating_matrix, axis=0)
        rating_matrix_adjusted = (
            rating_matrix - np.tile(item_bias, (rating_matrix.shape[0], 1))).copy()

        for j in range(rating_matrix.shape[1]):
            top_k_items = np.argsort(similarity_matrix[:, j])[:-k-1:-1]
            predictions[:, j] = np.dot(similarity_matrix[top_k_items, j].T, rating_matrix_adjusted[:,
                                       top_k_items].T) / np.sum(np.abs(similarity_matrix[top_k_items, j]))

        predictions += np.tile(item_bias, (rating_matrix.shape[0], 1))

    return predictions


def calculate_mse(predictions, actual_ratings):
    """
    计算预测评分矩阵和实际评分矩阵之间的均方误差（MSE）。

    Args:
        predictions (numpy.ndarray): 预测评分矩阵
        actual_ratings (numpy.ndarray): 实际评分矩阵

    Returns:
        float: 均方误差（MSE）
    """
    # 忽略评分为零的项
    non_zero_indices = actual_ratings.nonzero()
    predictions = predictions[non_zero_indices].flatten()
    actual_ratings = actual_ratings[non_zero_indices].flatten()

    return mean_squared_error(predictions, actual_ratings)


def als(ratings_matrix, num_features=40, user_regularization=0, item_regularization=0, iterations=10):
    """
    利用交替最小二乘法（ALS）算法计算用户和物品的特征向量。

    参数:
        ratings_matrix (array-like): 用户-物品评分矩阵。
        num_features (int, optional): 特征向量的维数，默认为40。
        user_regularization (float, optional): 用户正则化参数，用于防止过拟合，默认为0。
        item_regularization (float, optional): 物品正则化参数，同样用于防止过拟合，默认为0。
        iterations (int, optional): 迭代次数，默认为10。

    返回:
        array-like: 用户对物品的评分预测矩阵。
    """
    # 转置评分矩阵以便于处理
    ratings_matrix = ratings_matrix.T

    # 初始化用户和物品的特征向量矩阵
    user_vec = np.random.rand(ratings_matrix.shape[1], num_features).T
    res_vec = np.random.rand(ratings_matrix.shape[0], num_features).T

    # 迭代优化用户和物品的特征向量
    for i in range(iterations):
        for u in range(ratings_matrix.shape[1]):
            # 更新用户特征向量
            user_vec[:, u] = np.linalg.solve(np.dot(res_vec, res_vec.T) + user_regularization * np.eye(
                res_vec.shape[0]), np.dot(res_vec, ratings_matrix[:, u]))
        for r in range(ratings_matrix.shape[0]):
            # 更新物品特征向量
            res_vec[:, r] = np.linalg.solve(np.dot(user_vec, user_vec.T) + item_regularization * np.eye(
                user_vec.shape[0]), np.dot(user_vec, ratings_matrix[r, :].T))

    # 计算最终的预测评分矩阵
    return np.dot(res_vec.T, user_vec).T


def sgd_with_bias_correction(rating_matrix, num_features=40, user_bias_reg=0.01, item_bias_reg=0.01,
                             user_vector_reg=0.01, item_vector_reg=0.01, learning_rate=1e-3, iterations=200):
    """
    使用带偏差校正的随机梯度下降（SGD）进行矩阵分解，学习用户和物品的偏差以及潜在向量。

    Args:
        rating_matrix (numpy.ndarray): 用户-物品评分矩阵
        num_features (int): 潜在特征的数量
        user_bias_reg (float): 用户偏差的正则化参数
        item_bias_reg (float): 物品偏差的正则化参数
        user_vector_reg (float): 用户向量的正则化参数
        item_vector_reg (float): 物品向量的正则化参数
        learning_rate (float): 学习率
        iterations (int): 迭代次数

    Returns:
        tuple: 预测评分矩阵，误差数组，用户潜在向量，物品潜在向量
    """
    num_users, num_items = rating_matrix.shape
    error_array = np.zeros(iterations)

    global_bias = np.mean(rating_matrix[np.where(rating_matrix != 0)])
    user_bias = 0.1 * (2 * (np.random.rand(num_users) - 1))
    item_bias = 0.1 * (2 * (np.random.rand(num_items) - 1))
    user_vectors = 0.1 * (2 * (np.random.rand(num_users, num_features) - 1))
    item_vectors = 0.1 * (2 * (np.random.rand(num_items, num_features) - 1))

    training_indices = np.arange(rating_matrix.nonzero()[0].shape[0])
    np.random.shuffle(training_indices)
    non_zero_user_indices = rating_matrix.nonzero()[0]
    non_zero_item_indices = rating_matrix.nonzero()[1]

    for iteration in range(iterations):
        learning_rate = 1.0 / (100 + 0.01 * iteration)
        np.random.shuffle(training_indices)

        temp_error_array = np.zeros(len(training_indices))

        for datapoint_idx, idx in enumerate(training_indices):
            user = non_zero_user_indices[idx]
            item = non_zero_item_indices[idx]

            prediction = global_bias + \
                user_bias[user] + item_bias[item] + \
                np.dot(user_vectors[user, :], item_vectors[item, :].T)
            error = rating_matrix[user, item] - prediction
            temp_error_array[datapoint_idx] += error**2

            if iteration > 0:
                user_bias[user] += learning_rate * \
                    (error - user_bias_reg * user_bias[user])
                item_bias[item] += learning_rate * \
                    (error - item_bias_reg * item_bias[item])
                user_vectors[user, :] += learning_rate * \
                    (error * item_vectors[item, :] -
                     user_vector_reg * user_vectors[user, :])
                item_vectors[item, :] += learning_rate * \
                    (error * user_vectors[user, :] -
                     item_vector_reg * item_vectors[item, :])

        error_array[iteration] = np.mean(temp_error_array)

    predictions = global_bias + \
        user_bias[:, np.newaxis] + item_bias[np.newaxis, :] + \
        np.dot(user_vectors, item_vectors.T)
    predictions[predictions > 5] = 5
    predictions[predictions < 0] = 0

    return predictions, error_array, user_vectors, item_vectors


def concatenate_user_item_vectors(user_vectors, item_vectors, rating_matrix):
    """
    将用户向量和物品向量连接起来，并返回一个包含非零评分的矩阵。

    Args:
        user_vectors (numpy.ndarray): 用户潜在向量矩阵 (num_users x num_features)
        item_vectors (numpy.ndarray): 物品潜在向量矩阵 (num_items x num_features)
        rating_matrix (numpy.ndarray): 用户-物品评分矩阵

    Returns:
        numpy.ndarray: 包含用户向量、物品向量和对应评分的矩阵
    """
    non_zero_indices = rating_matrix.nonzero()
    user_vectors_non_zero = user_vectors[non_zero_indices[0]]
    item_vectors_non_zero = item_vectors[non_zero_indices[1]]
    ratings_non_zero = rating_matrix[non_zero_indices].reshape(-1, 1)

    concatenated_matrix = np.concatenate(
        (user_vectors_non_zero, item_vectors_non_zero, ratings_non_zero), axis=1)

    return concatenated_matrix
