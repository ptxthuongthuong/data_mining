from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

class ID3DecisionTree:
    def __init__(self, criterion='entropy'):  # Thêm tham số 'criterion'
        self.model = None
        self.feature_names = []
        self.class_names = []
        self.criterion = criterion  # Lưu giá trị 'criterion' vào thuộc tính

    def fit(self, X, y, feature_names=None):
        # Sử dụng criterion trong khởi tạo DecisionTreeClassifier
        self.model = DecisionTreeClassifier(criterion=self.criterion)  # Sử dụng criterion đã lưu
        self.model.fit(X, y)

        if feature_names is not None:
            self.feature_names = feature_names

        self.class_names = sorted(set(y))

    def predict(self, X):
        return self.model.predict(X)

    def export_tree(self):
        from sklearn.tree import export_text
        return export_text(self.model, feature_names=self.feature_names)

def visualize_tree(self, X, y):
    if not self.feature_names or not self.class_names:
        raise ValueError("Feature names and class names must be set before visualization.")

    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(self.model, 
              feature_names=self.feature_names,
              class_names=list(map(str, self.class_names)),
              filled=True, rounded=True, ax=ax)
    return fig
