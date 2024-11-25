import pandas as pd
import numpy as np

def apply_laplace_smoothing(df, target_column, input_data):
    """
    Thực hiện thuật toán Naive Bayes với Laplace smoothing.
    
    Args:
        df (pd.DataFrame): Dataset đầu vào.
        target_column (str): Cột mục tiêu.
        input_data (dict): Dữ liệu đầu vào để phân lớp, dạng {column_name: value}.
    
    Returns:
        str: Kết quả phân lớp.
    """
    # Lọc chỉ những cột được chọn
    df = df[list(input_data.keys()) + [target_column]]
    
    # Tính xác suất tiên nghiệm (prior probability)
    target_values = df[target_column].unique()
    priors = {value: len(df[df[target_column] == value]) / len(df) for value in target_values}
    
    # Tính xác suất có điều kiện (likelihood) với Laplace smoothing
    likelihoods = {}
    for col, value in input_data.items():
        likelihoods[col] = {}
        for target_value in target_values:
            subset = df[df[target_column] == target_value]
            # Tính xác suất có điều kiện với Laplace smoothing
            # Thêm 1 vào số lượng mẫu trong mỗi lớp và tổng số giá trị có thể của cột
            total_count = len(subset)
            count_value = len(subset[subset[col] == value])
            likelihoods[col][target_value] = (count_value + 1) / (total_count + len(df[col].unique()))  # Thêm 1 vào mẫu và cộng thêm số giá trị khác
    
    # Tính xác suất posterior cho mỗi lớp
    posteriors = {}
    for target_value in target_values:
        posteriors[target_value] = priors[target_value]
        for col in input_data:
            posteriors[target_value] *= likelihoods[col].get(target_value, 1e-6)  # Tránh nhân với 0
    
    # Chọn lớp có xác suất posterior cao nhất
    predicted_class = max(posteriors, key=posteriors.get)
    return predicted_class
def apply_naive_bayes(df, target_column, input_data):
    """
    Thực hiện thuật toán Naive Bayes mà không sử dụng Laplace smoothing.
    
    Args:
        df (pd.DataFrame): Dataset đầu vào.
        target_column (str): Cột mục tiêu.
        input_data (dict): Dữ liệu đầu vào để phân lớp, dạng {column_name: value}.
    
    Returns:
        str: Kết quả phân lớp.
    """
    # Lọc chỉ những cột được chọn
    df = df[list(input_data.keys()) + [target_column]]
    
    # Tính xác suất tiên nghiệm (prior probability)
    target_values = df[target_column].unique()
    priors = {value: len(df[df[target_column] == value]) / len(df) for value in target_values}
    
    # Tính xác suất có điều kiện (likelihood) mà không sử dụng Laplace smoothing
    likelihoods = {}
    for col, value in input_data.items():
        likelihoods[col] = {}
        for target_value in target_values:
            subset = df[df[target_column] == target_value]
            # Tính xác suất có điều kiện mà không cần Laplace smoothing
            count_value = len(subset[subset[col] == value])
            likelihoods[col][target_value] = count_value / len(subset) if len(subset) > 0 else 0  # Nếu không có giá trị nào, gán là 0
    
    # Tính xác suất posterior cho mỗi lớp
    posteriors = {}
    for target_value in target_values:
        posteriors[target_value] = priors[target_value]
        for col in input_data:
            posteriors[target_value] *= likelihoods[col].get(target_value, 1e-6)  # Tránh nhân với 0
    
    # Chọn lớp có xác suất posterior cao nhất
    predicted_class = max(posteriors, key=posteriors.get)
    return predicted_class
def naive_bayes_classifier(df, target_column, input_data, use_laplace=False):
    """
    Thực hiện thuật toán Naive Bayes, có thể chọn dùng Laplace smoothing.
    
    Args:
        df (pd.DataFrame): Dataset đầu vào.
        target_column (str): Cột mục tiêu.
        input_data (dict): Dữ liệu đầu vào để phân lớp, dạng {column_name: value}.
        use_laplace (bool): Flag để chọn sử dụng Laplace smoothing hay không.
    
    Returns:
        str: Kết quả phân lớp.
    """
    if use_laplace:
        return apply_laplace_smoothing(df, target_column, input_data)
    else:
        return apply_naive_bayes(df, target_column, input_data)
