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
def upper_approximation(X, equivalence_classes):
    upper = []
    for eq_class in equivalence_classes:
        if not eq_class[eq_class['decision_attribute'] == X['decision_attribute']].empty:
            upper.append(eq_class)
    return upper

# Hàm tìm xấp xỉ dưới của tập X theo tập thuộc tính B
def lower_approximation(X, equivalence_classes):
    lower = []
    for eq_class in equivalence_classes:
        if eq_class['decision_attribute'].iloc[0] == X['decision_attribute']:
            lower.append(eq_class)
    return lower

# Hàm tìm vùng biên của tập X
def boundary_region(X, upper, lower):
    return [class_ for class_ in upper if class_ not in lower]

# Hàm tìm vùng ngoài của tập X
def outside_region(X, equivalence_classes, upper, lower):
    return [class_ for class_ in equivalence_classes if class_ not in upper]

# Hàm tìm các reducts (tập con các thuộc tính quan trọng nhất)
def find_reducts(dataset, condition_attributes, decision_attribute):
    reducts = []
    for col in condition_attributes:
        if dataset[col].nunique() > 1:
            reducts.append(col)
    return reducts

# Hàm tìm ma trận phân biệt giữa các đối tượng
def find_discriminant_matrix(dataset, condition_attributes):
    return pd.crosstab(dataset[condition_attributes[0]], dataset[condition_attributes[1]])

# Hàm tính độ phụ thuộc giữa các thuộc tính điều kiện và thuộc tính quyết định
def calculate_dependency(dataset, condition_attributes, decision_attribute):
    total = len(dataset)
    dependent = len(dataset[dataset[condition_attributes].eq(dataset[decision_attribute], axis=0).all(axis=1)])
    return dependent / total
