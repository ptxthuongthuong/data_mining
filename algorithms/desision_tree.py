from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
import numpy as np
import graphviz

class ID3DecisionTree:
    def __init__(self, criterion='gini'):
        self.criterion = criterion
        self.model = DecisionTreeClassifier(criterion=criterion, random_state=42)
        self.feature_names = None
        self.class_names = None

    def train(self, X, y):
        self.feature_names = X.columns if hasattr(X, 'columns') else None
        self.class_names = np.unique(y).astype(str)
        self.model.fit(X, y)

    def get_tree_info(self, feature_names):
        tree = self.model.tree_

        def recurse(node, depth=0):
            node_info = {
                "node": node,
                "depth": depth,
                "samples": int(tree.n_node_samples[node]),
                "impurity": float(tree.impurity[node]),
            }
            if tree.feature[node] != -2:  # Không phải lá
                node_info["feature"] = feature_names[tree.feature[node]]
                node_info["threshold"] = float(tree.threshold[node])
            else:
                node_info["feature"] = "leaf"
                node_info["threshold"] = None
            return node_info

        nodes_info = []
        stack = [(0, 0)]
        while stack:
            node_id, depth = stack.pop(0)
            nodes_info.append(recurse(node_id, depth))
            if tree.feature[node_id] != -2:  # Không phải lá
                stack.append((tree.children_left[node_id], depth + 1))
                stack.append((tree.children_right[node_id], depth + 1))
        return nodes_info

    def export_tree(self, feature_names, class_names):
        dot_data = export_graphviz(
            self.model,
            out_file=None,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            special_characters=True
        )
        return graphviz.Source(dot_data)
