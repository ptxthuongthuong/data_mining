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
        self.nodes_cache = {}

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
        """Recursive ID3 algorithm implementation with independent child nodes for each branch."""
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

            # Create new node for each parent branch (to avoid sharing nodes)
            child_node = Node()  # Create a new child node for each parent branch
            child_node.value = feature_value
            
            # Recursively build the subtree for this new child node
            child.next = self._id3_recv(
                child_x_ids, 
                remaining_feature_ids, 
                child_node,  # Pass the new child node
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
    

def plot_custom_decision_tree(node, feature_names=None, class_names=None):
    """
    Vẽ cây quyết định với kích thước nhỏ gọn hơn
    """
    G = nx.DiGraph()
    pos = {}
    display_names = {}
    
    def add_nodes(current_node, parent=None, edge_label=None, depth=0, pos_x=0, parent_path=""):
        if current_node is None:
            return pos_x
            
        if current_node.value is not None:
            base_label = str(current_node.value)
            
            if not current_node.childs and current_node.class_distribution:
                class_dist = dict(current_node.class_distribution)
                base_label += f"\n{class_dist}"
                
            full_path = f"{parent_path}_{base_label}" if parent_path else base_label
            display_names[full_path] = base_label
        else:
            full_path = "Root"
            display_names[full_path] = "Root"
        
        G.add_node(full_path)
        
        # Điều chỉnh khoảng cách giữa các level
        level_spacing = 1.0  # Giảm xuống từ 2.0
        
        if parent is None:
            pos[full_path] = (0, 0)
        else:
            # Điều chỉnh khoảng cách ngang giữa các node
            sibling_spacing = 1.2  # Giảm xuống từ 2.0
            pos[full_path] = (pos_x * sibling_spacing, -depth * level_spacing)
        
        if parent is not None:
            G.add_edge(parent, full_path, label=edge_label or '')
        
        if current_node.childs:
            num_children = len(current_node.childs)
            # Điều chỉnh khoảng cách giữa các node con
            child_spacing = 0.6  # Giảm xuống để các node con gần nhau hơn
            start_pos = pos_x - (num_children - 1) * child_spacing / 2
            
            for i, child in enumerate(current_node.childs):
                child_pos = start_pos + i * child_spacing
                add_nodes(child.next, full_path, child.value, 
                         depth + 1, child_pos, full_path)
        
        return pos_x
    
    add_nodes(node)
    
    # Tính toán kích thước của đồ thị
    all_x = [coord[0] for coord in pos.values()]
    all_y = [coord[1] for coord in pos.values()]
    width = max(all_x) - min(all_x) + 1
    height = max(all_y) - min(all_y) + 1
    
    # Điều chỉnh kích thước figure
    plt.figure(figsize=(width * 1.5, height * 1.5))  # Giảm hệ số nhân xuống
    
    # Điều chỉnh kích thước node và font
    node_size = 3000  # Giảm kích thước node
    font_size = 7     # Giảm kích thước font
    
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=node_size, font_size=font_size, font_weight='bold',
            font_color='black', arrows=True,
            labels=display_names)
    
    # Điều chỉnh kích thước font của nhãn cạnh
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=font_size)
    
    plt.title("Decision Tree Visualization")
    plt.axis('off')
    plt.tight_layout()  # Thêm tight_layout để tối ưu không gian
    
    return plt