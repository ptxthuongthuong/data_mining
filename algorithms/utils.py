import pandas as pd
from sklearn.preprocessing import LabelEncoder

def update_dataset_info(filepath):
    # Đọc dataset
    df = pd.read_excel(filepath)

    # Làm sạch khoảng trắng trong tên cột
    df.columns = df.columns.str.strip()

    # Loại bỏ các khoảng trắng dư thừa trong dữ liệu của từng cột (nếu là chuỗi)
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].str.strip()

    # Kiểm tra giá trị null và thay thế nếu cần
    df.fillna("Unknown", inplace=True)

    # Trả về thông tin cột và giá trị khác biệt
    columns = df.columns.tolist()
    unique_values = {col: df[col].unique().tolist() for col in columns}
    return {"columns": columns, "unique_values": unique_values}

def encode_data(df):
    """
    Mã hóa dữ liệu sử dụng Label Encoding cho tất cả các cột chuỗi.
    """
    le = LabelEncoder()
    encoded_df = df.copy()  # Tạo bản sao để tránh warning
    for col in encoded_df.select_dtypes(include=['object', 'category']).columns:
        encoded_df.loc[:, col] = le.fit_transform(encoded_df[col])
    return encoded_df, le


