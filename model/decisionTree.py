from __future__ import print_function
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class TreeNode(object):
    def __init__(self, ids=None, children=[], entropy=0, depth=0, list_check_attribute=[]):
        self.ids = ids  # list row index in node
        self.entropy = entropy  # entropy
        self.depth = depth  # depth from root to current node
        self.split_attribute = None  # split_attribute if not leaf node
        self.children = children  # list children node
        self.order = None  # split_attribute order
        self.label = None  # label of node if current node is leaf node
        self.list_check_attribute = list_check_attribute

    def set_properties(self, split_attribute, order):
        self.split_attribute = split_attribute
        self.order = order

    def set_label(self, label):
        self.label = label


def entropy(freq):
    # remove prob 0
    freq_0 = freq[np.array(freq).nonzero()[0]]
    prob_0 = freq_0 / float(freq_0.sum())
    return -np.sum(prob_0 * np.log(prob_0))


class DecisionTreeID3(object):
    def __init__(self, max_depth=10, min_samples_split=2, min_gain=1e-4):
        self.root = None  # root node
        self.max_depth = max_depth  # max depth from root to current node
        self.min_samples_split = min_samples_split
        self.Ntrain = 0
        self.min_gain = min_gain  # ngưỡng entropy

    def fit(self, data, target):
        self.Ntrain = data.count()[0]
        self.data = data
        self.attributes = list(data)
        self.target = target
        self.labels = target.unique()

        ids = range(self.Ntrain)
        self.root = TreeNode(ids=ids, entropy=self._entropy(ids), depth=0, list_check_attribute=self.attributes)
        queue = [self.root]
        while queue:
            node = queue.pop()
            if node.depth < self.max_depth or node.entropy < self.min_gain:
                node.children = self._split(node)
                if not node.children:  # leaf node
                    self._set_label(node)
                queue += node.children
            else:
                self._set_label(node)

    def _entropy(self, ids):
        # calculate entropy of a node with index ids
        if len(ids) == 0: return 0
        ids = [i + 1 for i in ids]  # panda series index starts from 1
        freq = np.array(self.target[ids].value_counts())
        return entropy(freq)

    def _set_label(self, node):
        # find label for a node if it is a leaf
        # simply chose by major voting
        target_ids = [i + 1 for i in node.ids]  # target is a series variable
        node.set_label(self.target[target_ids].mode()[0])  # most frequent label

    def _split(self, node):
        ids = node.ids
        best_gain = 0
        best_splits = []
        best_attribute = None
        order = None
        sub_data = self.data.iloc[ids, :]
        for i, att in enumerate(node.list_check_attribute):
            values = self.data.iloc[ids, i].tolist()
            targets = self.target.tolist()
            values, targets = zip(*sorted(zip(values, targets)))
            splits = []
            order_result = []
            list = []
            for j in range(0, len(values)):
                if j > 0 and targets[j] != targets[j - 1]:
                    sub_ids = []
                    for check_val in list:
                        sub_ids += sub_data.index[sub_data[att] == check_val].unique().tolist()
                    splits.append([sub_id - 1 for sub_id in sub_ids])
                    if len(list) > 0:
                        order_result.append([min(list), max(list)])
                    list = []
                else:
                    list.append(values[j])
            # don't split if a node has too small number of points
            if min(map(len, splits)) < self.min_samples_split: continue
            # information gain
            HxS = 0
            for split in splits:
                HxS += len(split) * self._entropy(split) / len(ids)
            gain = node.entropy - HxS
            if gain < self.min_gain: continue  # stop if small gain
            if gain > best_gain:
                best_gain = gain
                best_splits = splits
                best_attribute = att
                order = order_result
        node.set_properties(best_attribute, order)
        child_nodes = [TreeNode(ids=split,
                                entropy=self._entropy(split), depth=node.depth + 1,
                                list_check_attribute=node.list_check_attribute.remove(best_attribute)) for split in
                       best_splits]
        return child_nodes

    def predict(self, new_data):
        npoints = new_data.count()[0]
        labels = [None] * npoints
        for n in range(npoints):
            x = new_data.iloc[n, :]  # one point
            # start from root and recursively travel if not meet a leaf
            node = self.root
            while node.children:
                for i, j in enumerate(node.order):
                    if x[node.split_attribute] >= j[0] and x[node.split_attribute] <= j[1]:
                        node = node.children[i]
                        break
            labels[n] = node.label
        return labels
