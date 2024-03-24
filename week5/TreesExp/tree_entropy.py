import numpy as np
import pandas as pd
def entropy(subset):
    value_counts = subset.value_counts()
    probabilities = value_counts / len(subset)
    return -np.sum(probabilities * np.log2(probabilities))


def informationGain(dataset, feature):
    total_entropy = entropy(dataset['target'])
    sum_parts_entropy = 0
    # subsets = dataset.groupby(feature)
    # for subset in subsets:
    #     sum_parts_entropy += np.sum(len(subset) * entropy(subset['target']))

    subsets = dataset.groupby(feature)
    for _, subset in subsets:
        sum_parts_entropy += len(subset) * entropy(subset['target'])

    return total_entropy - (sum_parts_entropy/len(dataset))

class TreeNode:
    def __init__(self, name='root', is_leaf=False):
        self.name = name
        self.is_leaf = is_leaf
        self.children = {}

    def add_child(self, value, node):
        self.children[value] = node

    def __str__(self, level=0):
        ret = "|  " * level + repr(self.name) + "\n"
        for value, subtree in self.children.items():
            ret += subtree.__str__(level + 1)
        return ret

    def __repr__(self):
        return '<' + self.name + '>'
def id3_tree(dataset, attributes):
    # Base cases for recursion
    if len(np.unique(dataset['target'])) == 1:
        return TreeNode(name=np.unique(dataset['target'])[0], is_leaf=True)
    elif len(attributes) == 0:
        return TreeNode(name=dataset['target'].value_counts().idxmax(), is_leaf=True)
    else:
        gains = [informationGain(dataset, attr) for attr in attributes]
        best_attr_index = np.argmax(gains)
        best_attr = attributes[best_attr_index]

        node = TreeNode(name=best_attr)
        new_attributes = attributes[:]
        new_attributes.remove(best_attr)
        for value in dataset[best_attr]:
            subset = dataset[dataset[best_attr] == value].drop(columns=[best_attr])
            child_node = id3_tree(subset, new_attributes)
            node.add_child(value, child_node)

        return node

data = {
    'COMS2': ['A', 'C', 'C', 'B', 'B', 'B', 'C', 'A', 'B'],
    'doing_labs': ['N', 'Y', 'N', 'Y', 'N', 'Y', 'Y', 'N', 'Y'],
    'doing_tuts': ['Y', 'N', 'Y', 'Y', 'N', 'N', 'N', 'N', 'N'],
    'target': ['Pass', 'Fail', 'Pass', 'Pass', 'Fail', 'Pass', 'Pass', 'Fail', 'Pass']
}

# Create the DataFrame
dataset = pd.DataFrame(data)

tree = id3_tree(dataset, attributes = dataset.columns[:-1].tolist())

print(tree)
