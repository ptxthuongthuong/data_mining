import pandas as pd
from itertools import combinations

def apriori_algorithm(transactions, min_support=0.1, min_confidence=0.5):
    """
    Thực hiện thuật toán Apriori để tìm các bộ itemset phổ biến và các quy tắc kết hợp.
    
    Args:
        transactions (list of list): Danh sách các giao dịch, mỗi giao dịch là một danh sách các item.
        min_support (float): Ngưỡng hỗ trợ tối thiểu để itemset được coi là phổ biến.
        min_confidence (float): Ngưỡng độ tin cậy tối thiểu để quy tắc kết hợp được chấp nhận.
    
    Returns:
        dict: Bao gồm 'frequent_itemsets' và 'rules' tìm được từ thuật toán Apriori.
    """
    # Bước 1: Tạo itemset 1 và tính toán hỗ trợ
    item_support = {}
    for transaction in transactions:
        for item in transaction:
            item_support[frozenset([item])] = item_support.get(frozenset([item]), 0) + 1

    # Chuyển sang dạng tỷ lệ (hỗ trợ)
    total_transactions = len(transactions)
    item_support = {item: count / total_transactions for item, count in item_support.items()}

    # Lọc các itemset với hỗ trợ >= min_support
    frequent_itemsets = [itemset for itemset, support in item_support.items() if support >= min_support]
    frequent_itemsets_support = {itemset: item_support[itemset] for itemset in frequent_itemsets}

    # Bước 2: Tìm các itemset lớn hơn (k >= 2)
    k = 2
    while True:
        candidates = []
        for i in range(len(frequent_itemsets)):
            for j in range(i + 1, len(frequent_itemsets)):
                union_set = frequent_itemsets[i].union(frequent_itemsets[j])
                if len(union_set) == k:
                    candidates.append(union_set)

        # Tính hỗ trợ cho các itemset mới
        candidate_support = {}
        for candidate in candidates:
            candidate_support[frozenset(candidate)] = sum(1 for transaction in transactions if candidate.issubset(transaction)) / total_transactions

        # Lọc các itemset có hỗ trợ >= min_support
        candidates = [itemset for itemset, support in candidate_support.items() if support >= min_support]
        if not candidates:
            break

        frequent_itemsets.extend(candidates)
        frequent_itemsets_support.update({itemset: candidate_support[itemset] for itemset in candidates})
        k += 1

    # Bước 3: Tạo quy tắc kết hợp từ các itemset phổ biến
    rules = []
    for itemset in frequent_itemsets_support.keys():
        if len(itemset) >= 2:
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    antecedent_support = frequent_itemsets_support[antecedent]
                    rule_support = frequent_itemsets_support[itemset]
                    confidence = rule_support / antecedent_support if antecedent_support > 0 else 0
                    if confidence >= min_confidence:
                        rules.append({
                            'antecedent': set(antecedent),
                            'consequent': set(consequent),
                            'confidence': confidence
                        })

    return {
        'frequent_itemsets': [list(itemset) for itemset in frequent_itemsets_support.keys()],
        'rules': rules
    }
