# reduct.py (module giảm chiều)

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import pandas as pd



# Hàm chuẩn hóa dữ liệu
def preprocess_data(data, normalize=True):
    """
    Xử lý trước dữ liệu: chuyển đổi cột dạng ký tự sang số và chuẩn hóa nếu cần.
    """
    # Chuyển đổi dữ liệu dạng ký tự sang dạng số
    for col in data.select_dtypes(include='object'):
        data[col] = data[col].astype('category').cat.codes

    # Chuẩn hóa dữ liệu nếu cần
    if normalize:
        scaler = StandardScaler()
        data[data.columns] = scaler.fit_transform(data[data.columns])

    return data

# Hàm tính lớp tương đương
def equivalent_classes(df, columns):
    classes = df[columns].drop_duplicates()
    return classes

# Hàm tính xấp xỉ dưới
def lower_approximation(df, target_class, columns):
    classes = equivalent_classes(df, columns)
    # Kiểm tra nếu tất cả các giá trị của hàng thuộc lớp mục tiêu
    lower_approx = df[df[columns].apply(lambda row: all(row[col] in classes[target_class].values for col in columns), axis=1)]
    return lower_approx

# Hàm tính xấp xỉ trên
def upper_approximation(df, target_class, columns):
    classes = equivalent_classes(df, columns)
    # Kiểm tra nếu ít nhất một giá trị của hàng thuộc lớp mục tiêu
    upper_approx = df[df[columns].apply(lambda row: any(row[col] in classes[target_class].values for col in columns), axis=1)]
    return upper_approx

# Hàm tính vùng biên
def boundary_region(df, target_class, columns):
    lower = lower_approximation(df, target_class, columns)
    upper = upper_approximation(df, target_class, columns)
    boundary = upper[~upper.index.isin(lower.index)]  # Chỉ lấy các đối tượng không có trong lower
    return boundary

# Hàm tính vùng ngoài
def outside_region(df, target_class, columns):
    upper = upper_approximation(df, target_class, columns)
    outside = df[~df.index.isin(upper.index)]  # Chỉ lấy các đối tượng không có trong upper
    return outside





def reduce_dimensions(df, columns, n_components):
    # Chọn các cột dữ liệu để giảm chiều
    data = df[columns]
    
    # Chuyển dữ liệu thành dạng số nếu cần thiết
    data = pd.get_dummies(data)
    
    # Áp dụng PCA để giảm chiều
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    
    # Chuyển lại dữ liệu thành DataFrame
    reduced_df = pd.DataFrame(reduced_data, columns=[f"Component {i+1}" for i in range(n_components)])
    
    # Giải thích phương sai của các thành phần chính
    explained_variance = pca.explained_variance_ratio_
    
    return reduced_df, explained_variance


