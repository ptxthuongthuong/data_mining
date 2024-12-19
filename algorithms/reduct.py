# reduct.py (module giảm chiều)

import pandas as pd
import numpy as np

# Hàm tìm tập X có cùng một thuộc tính quyết định mà người dùng đã chọn
def find_decision_class(dataset, decision_attribute, target_value):
    return dataset[dataset[decision_attribute] == target_value]

# Hàm tìm các lớp tương đương có cùng giá trị của tập thuộc tính B
def find_equivalence_classes(dataset, condition_attributes):
    classes = dataset[condition_attributes].drop_duplicates()
    equivalence_classes = []
    for _, row in classes.iterrows():
        equivalence_classes.append(dataset[(dataset[condition_attributes] == row).all(axis=1)])
    return equivalence_classes

# Hàm tìm xấp xỉ trên của tập X theo tập thuộc tính B
def upper_approximation(X, equivalence_classes, decision_column):
    """
    Xấp xỉ trên bao gồm tất cả các lớp tương đương có ít nhất một phần tử thuộc X
    """
    upper = []
    target_value = X[decision_column].iloc[0]
    
    for eq_class in equivalence_classes:
        # Kiểm tra xem lớp tương đương có giao với X không
        if any(index in X.index for index in eq_class.index):
            upper.append(eq_class)
            
    return upper

def lower_approximation(X, equivalence_classes, decision_column):
    """
    Xấp xỉ dưới bao gồm các lớp tương đương mà toàn bộ phần tử đều thuộc X
    """
    lower = []
    target_value = X[decision_column].iloc[0]
    
    for eq_class in equivalence_classes:
        # Kiểm tra xem tất cả các phần tử trong lớp tương đương có thuộc X không
        if all(index in X.index for index in eq_class.index):
            lower.append(eq_class)
            
    return lower


# Hàm tìm vùng biên của tập X
def boundary_region(X, upper, lower):
    boundary = []
    for class_ in upper:
        if not any(class_.equals(lower_class) for lower_class in lower):  # Kiểm tra nếu không có lớp nào trong lower tương đương với class_
            boundary.append(class_)
    return boundary


# Hàm tìm vùng ngoài của tập X
def outside_region(X, equivalence_classes, upper, lower):
    outside = []
    for class_ in equivalence_classes:
        if not any(class_.equals(upper_class) for upper_class in upper):  # Kiểm tra nếu class_ không có trong upper
            outside.append(class_)
    return outside


def create_discernibility_matrix(dataset, condition_attributes, decision_column):
    """
    Tạo ma trận phân biệt dựa trên các thuộc tính điều kiện
    """
    n = len(dataset)
    matrix = [[set() for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            # Chỉ xét các cặp có giá trị quyết định khác nhau
            if dataset.iloc[i][decision_column] != dataset.iloc[j][decision_column]:
                # Tìm các thuộc tính phân biệt được hai đối tượng
                discern_attrs = set()
                for attr in condition_attributes:
                    if dataset.iloc[i][attr] != dataset.iloc[j][attr]:
                        discern_attrs.add(attr)
                
                matrix[i][j] = discern_attrs
                matrix[j][i] = discern_attrs
    
    return matrix

def get_discernibility_function(matrix):
    """
    Tạo hàm phân biệt từ ma trận phân biệt
    """
    function = []
    n = len(matrix)
    
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j]:  # Nếu có thuộc tính phân biệt
                function.append(matrix[i][j])
    
    return function

def simplify_function(function):
    """
    Rút gọn hàm phân biệt bằng cách loại bỏ các tế trùng lặp và siêu tập
    """
    if not function:
        return []
    
    # Loại bỏ các tế trùng lặp
    unique_terms = list(set(frozenset(term) for term in function))
    unique_terms = [set(term) for term in unique_terms]
    
    # Loại bỏ các tế là siêu tập
    simplified = []
    for i, term1 in enumerate(unique_terms):
        is_minimal = True
        for j, term2 in enumerate(unique_terms):
            if i != j and term2.issubset(term1) and len(term2) < len(term1):
                is_minimal = False
                break
        if is_minimal:
            simplified.append(term1)
    
    return simplified

def calculate_dependency_degree(X, lower_approx):
    """
    Tính độ phụ thuộc của reduct
    """
    if not X.empty and lower_approx:
        lower_elements = set()
        for df in lower_approx:
            lower_elements.update(df.index)
        return len(lower_elements) / len(X)
    return 0

def find_reducts(dataset, condition_attributes, decision_column, X):
    """
    Tìm reduct sử dụng ma trận phân biệt và tính độ phụ thuộc
    
    Returns:
    - reduct: list các thuộc tính trong reduct
    - dependency_degree: độ phụ thuộc của reduct
    """
    # Tạo ma trận phân biệt
    matrix = create_discernibility_matrix(dataset, condition_attributes, decision_column)
    
    # Tạo hàm phân biệt
    function = get_discernibility_function(matrix)
    
    # Rút gọn hàm phân biệt
    simplified = simplify_function(function)
    
    # Nếu không tìm thấy reduct
    if not simplified:
        return condition_attributes, 0
    
    # Chọn reduct nhỏ nhất
    reduct = min(simplified, key=len)
    
    # Tính độ phụ thuộc của reduct
    reduct_list = list(reduct)
    IND = find_equivalence_classes(dataset, reduct_list)
    lower = lower_approximation(X, IND, decision_column)
    dependency_degree = calculate_dependency_degree(X, lower)
    
    return reduct_list, dependency_degree