import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data, normalize=True):
    """
    Xử lý trước dữ liệu: chuyển đổi cột dạng ký tự sang số và chuẩn hóa nếu cần.
    """
    # Chuyển đổi dữ liệu dạng ký tự sang dạng số
    for col in data.select_dtypes(include='object'):
        data[col] = data[col].astype('category').cat.codes

    # Chuẩn hóa dữ liệu
    if normalize:
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    return data

def calculate_euclidean_distance(point1, point2):
    """
    Tính khoảng cách Euclidean giữa hai điểm.
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))

def kmeans_clustering(data, k, max_iterations=100):
    """
    Thực hiện thuật toán K-Means Clustering.
    """
    # Chọn ngẫu nhiên các centroid ban đầu
    centroids = data.sample(n=k, random_state=42).values
    clusters = np.zeros(data.shape[0])

    for _ in range(max_iterations):
        # Gán mỗi điểm dữ liệu vào cụm gần nhất
        for i, point in enumerate(data.values):
            distances = [calculate_euclidean_distance(point, centroid) for centroid in centroids]
            clusters[i] = np.argmin(distances)

        # Cập nhật lại centroids
        new_centroids = []
        for i in range(k):
            cluster_points = data.values[clusters == i]
            if len(cluster_points) > 0:
                new_centroids.append(cluster_points.mean(axis=0))
            else:
                new_centroids.append(centroids[i])  # Giữ nguyên centroid cũ nếu cụm rỗng
        new_centroids = np.array(new_centroids)

        # Kiểm tra điều kiện dừng
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    # Tạo DataFrame kết quả
    data['Cluster'] = clusters.astype(int)
    return data, centroids
