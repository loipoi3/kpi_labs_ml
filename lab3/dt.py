import numpy as np
from binarytree import Node


class CustomDecisionTreeClassifier():
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        pass

    def _split_list_on_value_change(self, lst):
        results = []
        current_group = [lst[0]]
        for i in range(1, len(lst)):
            if lst[i] != lst[i - 1]:
                results.append((current_group[:], lst[i:]))
            current_group.append(lst[i])
        return results

    def _gini(self, lst):
        prob_p = np.sum(lst == 1) / len(lst)
        prob_n = np.sum(lst == 0) / len(lst)
        gini = prob_p ** 2 + prob_n ** 2
        return gini

    def _gini_impurity(self, y_lst):
        splited_y = self._split_list_on_value_change(y_lst)
        gini_impurity_lst = []
        for pair in splited_y:
            gini_f = self._gini(pair[0])
            gini_s = self._gini(pair[1])
            gini_impurity_f = 1 - gini_f
            gini_impurity_s = 1 - gini_s
            weighted_gini_impurited = len(pair[0]) / (len(pair[0]) + len(pair[1])) * gini_impurity_f + \
                                      len(pair[1]) / (len(pair[0]) + len(pair[1])) * gini_impurity_s
            gini_impurity_lst.append((weighted_gini_impurited, pair))
        return min(gini_impurity_lst, key=lambda x: x[0])

    def _built_bst(self, arr, depth=0):
        if len(arr) == 0 or len(list(set(arr.T[-1]))) == 1 or depth > 10:
            return None

        gini_impurity_all_features = []
        for i in range(arr.shape[1] - 1):
            sorted_indices = np.argsort(arr[:, i])
            sorted_array = arr[sorted_indices]
            gini_impurity_cortege = self._gini_impurity(sorted_array.T[-1])
            gini_impurity_all_features.append(gini_impurity_cortege)
        best_split_index = np.argmin([x[0] for x in gini_impurity_all_features])
        best_split = gini_impurity_all_features[best_split_index][1]
        sorted_indices = np.argsort(arr[:, best_split_index])
        sorted_array = arr[sorted_indices]
        spliter = (sorted_array[len(best_split[0]) - 1, best_split_index] + sorted_array[
            len(best_split[0]), best_split_index]) / 2

        node = Node(f"{best_split_index} {spliter}")

        mask = arr[:, best_split_index] < spliter
        left_arr = arr[mask]
        right_arr = arr[~mask]

        node.left = self._built_bst(left_arr, depth + 1)
        node.right = self._built_bst(right_arr, depth + 1)

        return node

    def _add_predict_to_nodes(self, node, arr):
        col_split = node.value.split(" ")
        mask = arr[:, int(col_split[0])] < float(col_split[1])
        left_arr = arr[mask]
        right_arr = arr[~mask]
        left_count_class_0 = np.sum(left_arr == 0)
        left_count_class_1 = np.sum(left_arr == 1)
        right_count_class_0 = np.sum(right_arr == 0)
        right_count_class_1 = np.sum(right_arr == 1)
        if node.left is not None:
            self._add_predict_to_nodes(node.left, left_arr)
        if node.right is not None:
            self._add_predict_to_nodes(node.right, right_arr)
        if node.left is None:
            node.left = Node(1 if left_count_class_1 > left_count_class_0 else 0)
        if node.right is None:
            node.right = Node(1 if right_count_class_1 > right_count_class_0 else 0)

    def fit(self, X, y):
        arr = np.concatenate((X, y[:, np.newaxis]), axis=1)
        self.bst = self._built_bst(arr)
        self._add_predict_to_nodes(self.bst, arr)

    def _predict(self, x, node):
        if not isinstance(node.value, int):
            col_split = node.value.split(" ")
            if x[int(col_split[0])] < float(col_split[1]):
                if not isinstance(node.left.value, int):
                    return self._predict(x, node.left)
                else:
                    return node.left.value
            else:
                if not isinstance(node.right.value, int):
                    return self._predict(x, node.right)
                else:
                    return node.right.value

    def predict(self, X):
        for x_i in X:
            print(self.bst)
            print(self._predict(x_i, self.bst))


dt = CustomDecisionTreeClassifier()
X = np.random.randint(1, 11, size=(10, 10))
y = np.random.randint(0, 2, size=(1, 10))[0]
print(X)
print(y)
print("="*50)
dt.fit(X, y)
x = np.random.randint(1, 11, size=(1, 10))
print(x)
dt.predict(x)
