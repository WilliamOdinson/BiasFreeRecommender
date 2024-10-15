import pandas as pd


def build_ensemble_df(svd, cosine,als, sgd, rf):
    df =pd.DataFrame(svd)
    df.columns = ['SVD']
    df['Cosine'] = cosine
    df['ALS'] = als
    df['SGD'] = sgd
    df['Random Forest'] = rf
    return df

def get_dtype_info(df):
    dtype_info = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
    return dtype_info



def build_ensemble_dataframe(svd_predictions, cosine_predictions, als_predictions, sgd_predictions, rf_predictions):
    """
    构建包含多种模型预测结果的DataFrame。

    Args:
        svd_predictions (numpy.ndarray): SVD模型的预测结果
        cosine_predictions (numpy.ndarray): 余弦相似性模型的预测结果
        als_predictions (numpy.ndarray): ALS模型的预测结果
        sgd_predictions (numpy.ndarray): SGD模型的预测结果
        rf_predictions (numpy.ndarray): 随机森林模型的预测结果

    Returns:
        pandas.DataFrame: 包含所有模型预测结果的DataFrame
    """
    ensemble_df = pd.DataFrame(svd_predictions, columns=['SVD'])
    ensemble_df['Cosine'] = cosine_predictions
    ensemble_df['ALS'] = als_predictions
    ensemble_df['SGD'] = sgd_predictions
    ensemble_df['Random Forest'] = rf_predictions
    
    return ensemble_df

def get_dataframe_dtypes(dataframe):
    """
    获取DataFrame中每一列的数据类型信息。

    Args:
        dataframe (pandas.DataFrame): 输入的DataFrame

    Returns:
        dict: 包含每一列名称及其数据类型的字典
    """
    dtype_info = {column: str(dtype) for column, dtype in zip(dataframe.columns, dataframe.dtypes)}
    
    return dtype_info
