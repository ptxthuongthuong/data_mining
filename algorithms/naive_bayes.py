def naive_bayes_classifier(df, target_column, input_data, use_laplace=False):
    # Kiểm tra xem có giá trị likelihood bằng 0 không
    def check_zero_likelihood(likelihoods):
        for feature, feature_likelihoods in likelihoods.items():
            for target_value, likelihood in feature_likelihoods.items():
                if likelihood == 0:
                    return f"Không phân lớp được vì feature '{feature}' cho class '{target_value}' có xác suất có điều kiện bằng 0. Vui lòng chuyển sang sử dụng **Laplace smoothing!**"
        return False

    # Nếu không dùng Laplace smoothing
    if not use_laplace:
        priors, likelihoods, posteriors = apply_naive_bayes(df, target_column, input_data)
        
        # Kiểm tra nếu có likelihood bằng 0
        zero_likelihood_check = check_zero_likelihood(likelihoods)
        if zero_likelihood_check:
            return zero_likelihood_check, priors, likelihoods, posteriors
    else:
        # Nếu dùng Laplace smoothing
        priors, likelihoods, posteriors = apply_laplace_smoothing(df, target_column, input_data)
    
    # Dự đoán lớp có xác suất posterior cao nhất
    predicted_class = max(posteriors, key=posteriors.get)
    
    # Trả về 4 giá trị: predicted_class, priors, likelihoods, posteriors
    return predicted_class, priors, likelihoods, posteriors


def apply_laplace_smoothing(df, target_column, input_data):
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
            count_value = len(subset[subset[col] == value])
            # Áp dụng Laplace smoothing (cộng thêm 1 vào số mẫu và tổng cộng vào số giá trị khác)
            likelihoods[col][target_value] = (count_value + 1) / (len(subset) + len(df[col].unique()))
    # Tính xác suất posterior cho mỗi lớp
    posteriors = {}
    for target_value in target_values:
        posteriors[target_value] = priors[target_value]
        for col in input_data:
            posteriors[target_value] *= likelihoods[col].get(target_value, 1e-6)  # Tránh nhân với 0
    # Trả về 3 giá trị: priors, likelihoods, posteriors
    return priors, likelihoods, posteriors

def apply_naive_bayes(df, target_column, input_data):
    # Lọc chỉ những cột được chọn
    df = df[list(input_data.keys()) + [target_column]]
    
    # Tính xác suất tiên nghiệm (prior probability)
    target_values = df[target_column].unique()
    priors = {value: len(df[df[target_column] == value]) / len(df) for value in target_values}
    
    # Tính xác suất có điều kiện (likelihood) mà không cần Laplace smoothing
    likelihoods = {}
    for col, value in input_data.items():
        likelihoods[col] = {}
        for target_value in target_values:
            subset = df[df[target_column] == target_value]
            count_value = len(subset[subset[col] == value])
            # Tính likelihood chính xác, có thể bằng 0
            likelihoods[col][target_value] = count_value / len(subset) if len(subset) > 0 else 0
    
    # Tính xác suất posterior cho mỗi lớp
    posteriors = {}
    for target_value in target_values:
        posteriors[target_value] = priors[target_value]
        for col in input_data:
            # Nếu likelihood bằng 0, posterior cũng sẽ bằng 0
            likelihood = likelihoods[col].get(target_value, 0)
            posteriors[target_value] *= likelihood if likelihood > 0 else 0
    
    return priors, likelihoods, posteriors

