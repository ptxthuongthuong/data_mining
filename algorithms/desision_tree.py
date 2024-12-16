import pandas as pd
import numpy as np
import math
from collections import Counter, deque
import networkx as nx
import matplotlib.pyplot as plt
class Node:
    """Contains the information of the node and another nodes of the Decision Tree."""
    def __init__(self):
        self.value = None
        self.next = None
        self.childs = None
        self.information_gain = None
        self.entropy = None
        self.gini = None
        self.class_distribution = None
        

class ID3DecisionTree:
    """Decision Tree Classifier using ID3 algorithm with support for Gini and Entropy."""
    def __init__(self, criterion='entropy'):
        self.X = None
        self.feature_names = None
        self.labels = None
        self.labelCategories = None
        self.node = None
        self.criterion = criterion
        self.class_names = None
        self.node_details = []

    def fit(self, X, y, feature_names):
        """Fit the decision tree model."""
        # Convert DataFrame to list of lists and list of labels
        self.X = X.values.tolist()
        self.labels = y.tolist()
        self.feature_names = feature_names
        self.labelCategories = list(set(self.labels))
        self.class_names = self.labelCategories
        self.model = self.node
        
        # Reset node details for each new tree
        self.node_details = []
        
        # Build the tree
        x_ids = [x for x in range(len(self.X))]
        feature_ids = [x for x in range(len(self.feature_names))]
        self.node = self._id3_recv(x_ids, feature_ids, None)

    def _calculate_entropy(self, labels):
        """Calculate entropy for a set of labels."""
        label_counts = Counter(labels)
        total = len(labels)
        entropy = 0
        for count in label_counts.values():
            prob = count / total
            entropy -= prob * math.log2(prob)
        return entropy

    def _calculate_gini(self, labels):
        """Calculate Gini impurity for a set of labels."""
        label_counts = Counter(labels)
        total = len(labels)
        gini = 1
        for count in label_counts.values():
            prob = count / total
            gini -= prob ** 2
        return gini

    def _get_feature_score(self, x_ids, feature_id):
        """Calculate information gain or Gini index based on selected criterion."""
        # Get labels and feature values for the selected samples
        labels = [self.labels[i] for i in x_ids]
        features = [self.X[i][feature_id] for i in x_ids]
        
        # Calculate initial impurity
        if self.criterion == 'entropy':
            initial_impurity = self._calculate_entropy(labels)
            score_func = self._calculate_entropy
        else:  # Gini
            initial_impurity = self._calculate_gini(labels)
            score_func = self._calculate_gini
        
        # Group by feature values
        feature_groups = {}
        for label, feature in zip(labels, features):
            if feature not in feature_groups:
                feature_groups[feature] = []
            feature_groups[feature].append(label)
        
        # Calculate weighted impurity
        weighted_impurity = 0
        for group, group_labels in feature_groups.items():
            weight = len(group_labels) / len(labels)
            weighted_impurity += weight * score_func(group_labels)
        
        # Information gain is the reduction in impurity
        score = initial_impurity - weighted_impurity
        return score, feature_groups

    def _get_feature_max_score(self, x_ids, feature_ids):
        """Find the best feature based on information gain or Gini index."""
        feature_scores = []
        feature_details = []
        
        for feature_id in feature_ids:
            score, feature_groups = self._get_feature_score(x_ids, feature_id)
            feature_scores.append(score)
            feature_details.append((score, feature_id, feature_groups))
        
        # Find feature with max score
        max_idx = feature_scores.index(max(feature_scores))
        best_score, best_feature_id, best_feature_groups = feature_details[max_idx]
        
        return (self.feature_names[best_feature_id], 
                best_feature_id, 
                best_score, 
                best_feature_groups)

    def _id3_recv(self, x_ids, feature_ids, node, parent_feature=None, parent_value=None):
        """Recursive ID3 algorithm implementation."""
        if not node:
            node = Node()

        # Get labels for current subset
        labels = [self.labels[i] for i in x_ids]
        
        # Class distribution for current node
        class_distribution = Counter(labels)

        # If all examples have the same class, create leaf node
        if len(set(labels)) == 1:
            node.value = labels[0]
            node.class_distribution = class_distribution
            # Record node details
            self.node_details.append({
                'parent_feature': parent_feature,
                'parent_value': parent_value,
                'node_value': node.value,
                'is_leaf': True,
                'class_distribution': dict(class_distribution),
                'entropy': self._calculate_entropy(labels) if self.criterion == 'entropy' else None,
                'gini': self._calculate_gini(labels) if self.criterion == 'gini' else None
            })
            return node

        # If no more features, return most common class
        if len(feature_ids) == 0:
            most_common_class = max(set(labels), key=labels.count)
            node.value = most_common_class
            node.class_distribution = class_distribution
            # Record node details
            self.node_details.append({
                'parent_feature': parent_feature,
                'parent_value': parent_value,
                'node_value': node.value,
                'is_leaf': True,
                'class_distribution': dict(class_distribution),
                'entropy': self._calculate_entropy(labels) if self.criterion == 'entropy' else None,
                'gini': self._calculate_gini(labels) if self.criterion == 'gini' else None
            })
            return node

        # Choose best feature
        best_feature_name, best_feature_id, best_score, feature_groups = self._get_feature_max_score(x_ids, feature_ids)
        
        # Set node properties
        node.value = best_feature_name
        node.information_gain = best_score
        node.entropy = self._calculate_entropy(labels) if self.criterion == 'entropy' else None
        node.gini = self._calculate_gini(labels) if self.criterion == 'gini' else None
        node.class_distribution = class_distribution
        node.childs = []

        # Record node details
        self.node_details.append({
            'parent_feature': parent_feature,
            'parent_value': parent_value,
            'node_value': best_feature_name,
            'is_leaf': False,
            'information_gain': best_score,
            'entropy': node.entropy,
            'gini': node.gini,
            'class_distribution': dict(class_distribution)
        })

        # Prepare for next recursion
        remaining_feature_ids = feature_ids.copy()
        remaining_feature_ids.remove(best_feature_id)

        # Create branches for each feature value
        for feature_value, value_labels in feature_groups.items():
            child = Node()
            child.value = feature_value
            node.childs.append(child)

            # Get sample ids for this feature value
            child_x_ids = [x_ids[i] for i in range(len(x_ids)) 
                           if self.X[x_ids[i]][best_feature_id] == feature_value]

            # Recursively build subtree
            child.next = self._id3_recv(
                child_x_ids, 
                remaining_feature_ids, 
                child.next, 
                parent_feature=best_feature_name,
                parent_value=feature_value
            )

        return node

    def export_node_details(self):
        """Export detailed node information as a DataFrame."""
        return pd.DataFrame(self.node_details)

    def export_tree(self):
        """Export tree structure as text."""
        if not self.node:
            return "Empty tree"
        
        tree_desc = []
        nodes = deque([self.node])
        while nodes:
            node = nodes.popleft()
            tree_desc.append(f"Node: {node.value}")
            
            if node.childs:
                for child in node.childs:
                    tree_desc.append(f"  Branch: {child.value}")
                    if child.next:
                        if isinstance(child.next, Node):
                            nodes.append(child.next)
                        else:
                            tree_desc.append(f"    Leaf: {child.next}")
        
        return '\n'.join(tree_desc)
    
# def plot_custom_decision_tree(node, feature_names=None, class_names=None):
#     """
#     Vẽ cây quyết định từ cấu trúc Node của ID3DecisionTree
    
#     Parameters:
#     - node: Nút gốc của cây
#     - feature_names: Danh sách tên các đặc trưng (tùy chọn)
#     - class_names: Danh sách tên các lớp (tùy chọn)
#     """
#     # Tạo đồ thị
#     G = nx.DiGraph()
    
#     # Thông tin vị trí các nút
#     pos = {}
    
#     # Hàm đệ quy để duyệt cây và thêm các nút
#     def add_nodes(current_node, parent=None, edge_label=None, depth=0, pos_x=0):
#         if current_node is None:
#             return pos_x
        
#         # Xác định nhãn nút
#         if current_node.value is not None:
#             node_label = str(current_node.value)
            
#             # Nếu là nút lá, thêm thông tin phân phối lớp
#             if not current_node.childs:
#                 if current_node.class_distribution:
#                     class_dist = dict(current_node.class_distribution)
#                     node_label += f"\n{class_dist}"
#         else:
#             node_label = "Root"
        
#         # Thêm nút vào đồ thị
#         G.add_node(node_label)
        
#         # Đặt vị trí của nút (sắp xếp theo chiều dọc)
#         pos[node_label] = (pos_x, -depth)  # Đảo ngược trục y để cây đi xuống
        
#         # Kết nối với nút cha nếu có
#         if parent is not None:
#             G.add_edge(parent, node_label, label=edge_label or '')
        
#         # Đệ quy với các nút con
#         if current_node.childs:
#             # Căn giữa các nút con
#             new_pos_x = pos_x - len(current_node.childs) // 2  # Căn giữa
#             for child in current_node.childs:
#                 new_pos_x = add_nodes(child.next, node_label, child.value, depth + 1, new_pos_x)
#                 new_pos_x += 1  # Tăng thêm một bước cho mỗi nút con để tránh chồng lấn
        
#         return pos_x
    
#     # Bắt đầu duyệt cây từ nút gốc
#     add_nodes(node)
    
#     # Vẽ đồ thị
#     plt.figure(figsize=(20, 10))
    
#     # Vẽ nút
#     nx.draw(G, pos, with_labels=True, node_color='lightblue', 
#             node_size=3000, font_size=8, font_weight='bold', font_color='black')
    
#     # Vẽ nhãn cạnh
#     edge_labels = nx.get_edge_attributes(G, 'label')
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
#     plt.title("Decision Tree Visualization")
#     plt.axis('off')
    
#     return plt

# def plot_custom_decision_tree(node, feature_names=None, class_names=None):
#     """
#     Vẽ cây quyết định từ cấu trúc Node của ID3DecisionTree
    
#     Parameters:
#     - node: Nút gốc của cây
#     - feature_names: Danh sách tên các đặc trưng (tùy chọn)
#     - class_names: Danh sách tên các lớp (tùy chọn)
#     """
#     # Tạo đồ thị
#     G = nx.DiGraph()
    
#     # Thông tin vị trí các nút
#     pos = {}
    
#     # Hàm đệ quy để duyệt cây và thêm các nút
#     def add_nodes(current_node, parent=None, edge_label=None, depth=0, pos_x=0):
#         if current_node is None:
#             return pos_x
        
#         # Xác định nhãn nút
#         if current_node.value is not None:
#             node_label = str(current_node.value)
            
#             # Nếu là nút lá, thêm thông tin phân phối lớp
#             if not current_node.childs:
#                 if current_node.class_distribution:
#                     class_dist = dict(current_node.class_distribution)
#                     node_label += f"\n{class_dist}"
#         else:
#             node_label = "Root"
        
#         # Thêm nút vào đồ thị
#         G.add_node(node_label)
        
#         # Đặt vị trí của nút (sắp xếp theo chiều dọc)
#         pos[node_label] = (pos_x, -depth)  # Đảo ngược trục y để cây đi xuống
        
#         # Kết nối với nút cha nếu có
#         if parent is not None:
#             G.add_edge(parent, node_label, label=edge_label or '')
        
#         # Đệ quy với các nút con
#         if current_node.childs:
#             # Căn giữa các nút con
#             new_pos_x = pos_x - len(current_node.childs) // 2  # Căn giữa
#             for child in current_node.childs:
#                 new_pos_x = add_nodes(child.next, node_label, child.value, depth + 1, new_pos_x)
#                 new_pos_x += 1  # Tăng thêm một bước cho mỗi nút con để tránh chồng lấn
        
#         return pos_x
    
#     # Bắt đầu duyệt cây từ nút gốc
#     add_nodes(node)
    
#     # Vẽ đồ thị
#     plt.figure(figsize=(12, 8))  # Giới hạn kích thước đồ thị
#     nx.draw(G, pos, with_labels=True, node_color='lightblue', 
#             node_size=3000, font_size=8, font_weight='bold', font_color='black')
    
#     # Vẽ nhãn cạnh
#     edge_labels = nx.get_edge_attributes(G, 'label')
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
#     # Tối ưu hóa bố cục để tránh chồng lấn
#     plt.tight_layout()
    
#     # Đảm bảo không có phần nào bị cắt
#     plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
#     plt.title("Decision Tree Visualization")
#     plt.axis('off')
    
#     return plt

def plot_custom_decision_tree(node, feature_names=None, class_names=None):
    """
    Vẽ cây quyết định từ cấu trúc Node của ID3DecisionTree
    
    Parameters:
    - node: Nút gốc của cây
    - feature_names: Danh sách tên các đặc trưng (tùy chọn)
    - class_names: Danh sách tên các lớp (tùy chọn)
    """
    # Tạo đồ thị
    G = nx.DiGraph()
    
    # Thông tin vị trí các nút
    pos = {}
    
    # Hàm đệ quy để duyệt cây và thêm các nút
    def add_nodes(current_node, parent=None, edge_label=None, depth=0, pos_x=0, level_widths=None):
        if current_node is None:
            return pos_x
        
        # Xác định nhãn nút
        if current_node.value is not None:
            node_label = str(current_node.value)
            
            # Nếu là nút lá, thêm thông tin phân phối lớp
            if not current_node.childs:
                if current_node.class_distribution:
                    class_dist = dict(current_node.class_distribution)
                    node_label += f"\n{class_dist}"
        else:
            node_label = "Root"
        
        # Thêm nút vào đồ thị
        G.add_node(node_label)
        
        # Đặt vị trí của nút (sắp xếp theo chiều dọc)
        pos[node_label] = (pos_x, -depth)  # Đảo ngược trục y để cây đi xuống
        
        # Cập nhật chiều rộng của từng cấp độ
        if level_widths is None:
            level_widths = {}
        
        if depth not in level_widths:
            level_widths[depth] = []
        
        level_widths[depth].append(pos_x)
        
        # Kết nối với nút cha nếu có
        if parent is not None:
            G.add_edge(parent, node_label, label=edge_label or '')
        
        # Đệ quy với các nút con
        if current_node.childs:
            # Căn giữa các nút con
            new_pos_x = pos_x - len(current_node.childs) // 2  # Căn giữa
            for child in current_node.childs:
                new_pos_x = add_nodes(child.next, node_label, child.value, depth + 1, new_pos_x, level_widths)
                new_pos_x += 1  # Tăng thêm một bước cho mỗi nút con để tránh chồng lấn
        
        return pos_x
    
    # Bắt đầu duyệt cây từ nút gốc
    add_nodes(node)
    
    # Điều chỉnh vị trí các nút con để căn giữa cây
    min_x = min([pos[node][0] for node in pos])
    max_x = max([pos[node][0] for node in pos])
    tree_width = max_x - min_x
    shift_x = -min_x + (tree_width / 2)  # Dịch cây vào giữa
    
    # Cập nhật lại vị trí các nút theo dịch chuyển đã tính
    for node_label in pos:
        pos[node_label] = (pos[node_label][0] + shift_x, pos[node_label][1])
    
    # Vẽ đồ thị
    plt.figure(figsize=(12, 8))  # Giới hạn kích thước đồ thị
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=3000, font_size=8, font_weight='bold', font_color='black')
    
    # Vẽ nhãn cạnh
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    # Tối ưu hóa bố cục để tránh chồng lấn
    plt.tight_layout()
    
    # Đảm bảo không có phần nào bị cắt
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    plt.title("Decision Tree Visualization")
    plt.axis('off')
    
    return plt
