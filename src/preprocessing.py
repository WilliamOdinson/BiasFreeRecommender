import re
import json
import openai
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, accuracy_score
from scipy import sparse
from scipy.sparse import csr_matrix, lil_matrix
from concurrent.futures import ThreadPoolExecutor


def get_top_m_num_reviews_for_city_and_business(df, m):
    """
    获取按城市和分类统计的商家数量，并返回前m个结果。

    Args:
        df (pd.DataFrame): 包含商家信息的数据框。
        m (int): 要返回的前m个结果的数量。

    Returns:
        pd.Series: 包含前m个按城市和分类统计的商家数量的Series。
    """
    # 创建business_city_count字典，用于存储城市和分类的商家数量
    business_city_count = {}
    n = len(df)

    # 遍历数据框的每一行
    for i in range(n):
        # 获取分类并拆分为列表
        categories = str(df.categories.iloc[i]).split(',')
        # 获取城市名称
        city = df.city.iloc[i]
        # 遍历每个分类
        for category in categories:
            key = (category, city)  # 创建分类和城市的元组
            # 如果元组不存在于字典中，则初始化为1，否则加1
            if key not in business_city_count.keys():
                business_city_count[key] = 1
            else:
                business_city_count[key] += 1

    # 将字典转换为Series并排序
    business_city_count_series = pd.Series(business_city_count)
    business_city_count_series.sort_values(ascending=False, inplace=True)

    return business_city_count_series[:m]


def get_clean_df(df, cols, min_user_review=30, min_res_review=0):
    """
    清洗数据并根据最小评论数量进行过滤

    参数:
        df (DataFrame): 原始数据框
        cols (list): 需要保留的列名列表
        min_user_review (int, 可选): 用户最少评论数，默认值为30
        min_res_review (int, 可选): 商家最少评论数，默认值为0

    返回:
        DataFrame: 清洗并过滤后的数据框
    """
    # 使用.copy()避免SettingWithCopyWarning
    df_new = df[cols].copy()
    # 删除任何包含缺失值的行
    df_new.dropna(axis=0, how='any', inplace=True)

    # 计算每个商家的评论数，并过滤评论数不满足条件的商家
    df_new[cols[1] +
           '_freq'] = df_new.groupby(cols[1])[cols[1]].transform('count')
    df_clean = df_new[df_new[cols[1] + '_freq'] >= min_res_review]

    # 计算每个用户的评论数，并过滤评论数不满足条件的用户
    df_clean[cols[0] +
             '_freq'] = df_clean.groupby(cols[0])[cols[0]].transform('count')
    df_clean_2 = df_clean[df_clean[cols[0] + '_freq'] >= min_user_review]

    return df_clean_2


def get_sparsity(sparse_matrix):
    """
    计算稀疏矩阵的稀疏度

    参数:
        sparse_matrix (csr_matrix): 稀疏矩阵

    返回:
        float: 稀疏度，表示为稀疏矩阵中非零元素的比例
    """
    density = sparse_matrix.nnz / \
        (sparse_matrix.shape[0] * sparse_matrix.shape[1])
    return 1 - density


def get_sparse_matrix(df):
    """
    将数据框转换为稀疏评分矩阵

    参数:
        df (DataFrame): 包含用户、商家和评分的数据框

    返回:
        csr_matrix: 稀疏评分矩阵
    """
    # 获取唯一用户ID列表
    unique_users = list(df['user_id'].unique())
    # 获取唯一商家ID列表
    unique_businesses = list(df['business_id'].unique())
    # 获取评分数据
    ratings = df['stars'].tolist()

    # 将用户ID映射到稀疏矩阵的行索引
    user_indices = pd.Categorical(df['user_id'], categories=unique_users).codes
    # 将商家ID映射到稀疏矩阵的列索引
    business_indices = pd.Categorical(
        df['business_id'], categories=unique_businesses).codes

    # 创建稀疏评分矩阵
    sparse_matrix = csr_matrix((ratings, (user_indices, business_indices)),
                               shape=(len(unique_users), len(unique_businesses)))
    return sparse_matrix


def train_val_test_split(sparse_matrix, num_review_val=2, num_review_test=2):
    """将稀疏矩阵划分为训练集、验证集和测试集

    Args:
        sparse_matrix (csr_matrix): 输入的稀疏矩阵
        num_review_val (int, optional): 每个用户用于验证的评论数. 默认为2.
        num_review_test (int, optional): 每个用户用于测试的评论数. 默认为2.

    Returns:
        tuple: 包含训练集、验证集和测试集的稀疏矩阵
    """
    # 获取非零元素的行索引和列索引
    nonzero_rows, nonzero_cols = sparse_matrix.nonzero()

    # 初始化训练集、验证集和测试集稀疏矩阵
    sparse_matrix_test = csr_matrix(sparse_matrix.shape)
    sparse_matrix_val = csr_matrix(sparse_matrix.shape)
    sparse_matrix_train = sparse_matrix.copy()

    num_users = sparse_matrix.shape[0]  # 用户数

    for user in range(num_users):
        # 获取该用户的所有评论索引
        user_review_indices = nonzero_cols[np.where(nonzero_rows == user)]

        # 打乱索引顺序
        np.random.shuffle(user_review_indices)

        # 划分测试集和验证集索引
        test_indices = user_review_indices[-num_review_test:]
        val_indices = user_review_indices[-(num_review_val +
                                            num_review_test):-num_review_test]

        # 将测试集和验证集的索引设置到对应的稀疏矩阵中
        sparse_matrix_test[user,
                           test_indices] = sparse_matrix[user, test_indices]
        sparse_matrix_val[user, val_indices] = sparse_matrix[user, val_indices]

        # 从训练集中移除验证集和测试集的索引
        sparse_matrix_train[user, test_indices] = 0
        sparse_matrix_train[user, val_indices] = 0

    # 重新创建训练集稀疏矩阵以去除零元素
    train_data = np.array(
        sparse_matrix_train[sparse_matrix_train.nonzero()])[0]
    train_rows = sparse_matrix_train.nonzero()[0]
    train_cols = sparse_matrix_train.nonzero()[1]
    matrix_size = sparse_matrix_train.shape

    sparse_matrix_train = csr_matrix(
        (train_data, (train_rows, train_cols)), shape=matrix_size)

    # 确保训练集、验证集和测试集没有重叠
    overlap_val_test = sparse_matrix_train.multiply(sparse_matrix_val)
    overlap_all = overlap_val_test.multiply(sparse_matrix_test)

    assert overlap_all.nnz == 0, "训练集、验证集和测试集之间存在重叠元素"

    return sparse_matrix_train, sparse_matrix_val, sparse_matrix_test


class NBFeatures(BaseEstimator):
    """朴素贝叶斯特征类

    Args:
        BaseEstimator (class): Scikit-learn的基础估计器类型
    """

    def __init__(self, alpha):
        """
        初始化函数，设置平滑参数。

        Args:
            alpha (float): 平滑参数，用于概率计算中的平滑处理，通常为1
        """
        self.alpha = alpha

    def adjust_features(self, x, r):
        """
        调整特征数据x，使用对数概率比r进行调整。

        Args:
            x (sparse matrix): 原始特征矩阵
            r (sparse matrix): 来自fit方法的对数比率矩阵

        Returns:
            sparse matrix: 调整后的特征矩阵
        """
        return x.multiply(r)

    def compute_class_prob(self, x, y_i, y):
        """
        计算指定类别y_i的条件概率。

        Args:
            x (sparse matrix): 特征数据
            y_i (int): 指定的类别
            y (array): 整个数据集的标签数组

        Returns:
            sparse matrix: 给定类别的条件概率
        """
        p = x[y == y_i].sum(0)
        return (p + self.alpha) / ((y == y_i).sum() + self.alpha)

    def fit(self, x, y=None):
        """
        计算每个特征的对数概率比，并以稀疏矩阵形式存储。

        Args:
            x (sparse matrix): 特征数据
            y (array, optional): 数据集的标签数组

        Returns:
            self: 返回自身对象，使得可以链式调用
        """
        self._r = sparse.csr_matrix(np.log(self.compute_class_prob(
            x, 1, y) / self.compute_class_prob(x, 0, y)))
        return self

    def transform(self, x):
        """
        应用朴素贝叶斯转换到原始特征矩阵x。

        Args:
            x (sparse matrix): 原始特征矩阵

        Returns:
            sparse matrix: 转换后的特征矩阵
        """
        x_nb = self.adjust_features(x, self._r)
        return x_nb


def get_coefs(word, *arr):
    """
    将从GloVe文件中读取的单词和其向量值转换为更易于处理的格式。

    Args:
        word (str): 从GloVe文件中读取的单词。
        *arr (str): 与word关联的向量数值，传入时为字符串，会被转换为浮点数数组。

    Returns:
        tuple: 包含单词和其对应的NumPy数组形式向量的元组。
    """
    try:
        return word, np.asarray(arr, dtype='float32')
    except ValueError:
        return word, None


def replace_abbreviations(text, abbreviation_dict):
    """
    替换文本中的缩写为全称

    参数:
        text (str): 输入的文本
        abbreviation_dict (dict): 缩写词典，键为缩写，值为全称列表

    返回:
        str: 替换后的文本
    """
    for abbr, full_forms in abbreviation_dict.items():
        # 选择第一个全称作为替换目标
        full_form = full_forms[0]
        # 使用正则表达式进行替换，确保匹配完整单词
        text = re.sub(r'\b' + re.escape(abbr) + r'\b', full_form, text)
    return text


def process_row(row, abbreviation_dict):
    """
    处理单行文本，替换其中的缩写为全称

    参数:
        row (str): 输入的单行文本
        abbreviation_dict (dict): 缩写词典，键为缩写，值为全称列表

    返回:
        str: 替换后的单行文本
    """
    return replace_abbreviations(row, abbreviation_dict)


def process_string(original_data):
    """处理字符串

    Args:
        original_data (str): 输入的原始数据

    Returns:
        dict: 包含原始数据和处理后响应的字典
    """
    if pd.isna(original_data):
        return {'original': original_data, 'response': ""}

    # 调用OpenAI API处理字符串
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        max_tokens=100,
        messages=[
            {"role": "system",
             "content": "I will give you some abbreviations and mistakes in the comments, please convert the following content into its original English form (non-abbreviated form), it may be internet slang. Your answer must really have this word (or several words). You only need to respond with the original form, no other content or explanations are allowed."},
            {"role": "user", "content": original_data},
        ]
    )

    # 提取并处理响应
    processed_data_content = response.choices[0].message.content
    processed_data_content = "|".join(processed_data_content.split("\n"))

    return {'original': original_data, 'response': processed_data_content}


def process_strings(strings_list):
    """处理字符串列表

    Args:
        strings_list (list): 输入的字符串列表

    Returns:
        list: 处理后的数据列表
    """
    processed_data = []
    # 使用ThreadPoolExecutor进行并行处理
    with ThreadPoolExecutor(max_workers=4) as executor:
        processed_data = list(executor.map(process_string, strings_list))
    return processed_data


def save_to_json(data_list, filename='../assets/contractions.json'):
    """保存数据到JSON文件

    Args:
        data_list (list): 处理后的数据列表
        filename (str, optional): 文件名，默认为'../assets/contractions.json'.
    """
    contractions_dict = {}
    for item in data_list:
        original = item['original']
        expanded = item['response']
        if original in contractions_dict:
            if expanded not in contractions_dict[original]:
                contractions_dict[original].append(expanded)
        else:
            contractions_dict[original] = [expanded]

    with open(filename, 'w') as json_file:
        json.dump(contractions_dict, json_file, indent=4)

    print(f"JSON文件已成功创建于 {filename}")


def Bert_preprocess(input_text, tokenizer):
    """
    使用指定的分词器（tokenizer）对输入文本进行预处理，以便用于BERT模型。

    参数:
        input_text (str): 需要处理的原始文本字符串。
        tokenizer: 用于文本处理的分词器对象。

    返回:
        dict: 包含编码后的输入数据，如输入ids、注意力遮罩等，格式为pytorch张量。

    详细说明:
        - add_special_tokens: 添加特殊令牌（如[CLS], [SEP]），这对于BERT模型是必需的。
        - padding: 将序列填充到一定的最大长度，确保所有的输入序列长度相同。
        - max_length: 设置序列的最大长度。如果序列长于此长度，将会被截断。
        - truncation: 如果序列长于最大长度，启用截断。
        - return_attention_mask: 返回注意力遮罩，用于区分实际内容和填充内容。
        - return_tensors: 指定返回数据的类型（在这里是PyTorch张量格式）。
    """
    return tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,  # 添加特殊令牌
        padding="max_length",     # 填充至最大长度
        max_length=512,           # 设定最大长度
        truncation=True,          # 进行截断
        return_attention_mask=True,  # 返回注意力遮罩
        return_tensors='pt'       # 返回pytorch张量
    )


def Bert_compute_batch_accuracy(logits, labels):
    """
    计算批次数据的准确率。

    参数：
        logits (numpy.ndarray): 模型输出的逻辑值，通常是每个类别的得分或概率预测。
        labels (numpy.ndarray): 真实的标签，通常是独热编码形式。

    返回：
        float: 计算得到的准确率。
    """
    preds = np.argmax(logits, axis=1).flatten()     # 从逻辑值中获取最大值的索引，作为预测结果
    truth = np.argmax(labels, axis=1).flatten()     # 从真实标签中获取最大值的索引，作为真实结果
    return accuracy_score(truth, preds)             # 计算并返回准确率


if __name__ == '__main__':
    # 初始化OpenAI客户端
    client = openai.OpenAI()
    # 替换为你的实际字符串列表
    strings_list = ["can't", "won't", "they're"]
    processed_data = process_strings(strings_list)
    save_to_json(processed_data)
