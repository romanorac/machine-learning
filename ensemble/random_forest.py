import numpy as np

import decision_tree


def bootstrap(x, y):
    # generate len(x) random numbers with replacement from 0 to range(len(x))
    bag_indices = np.random.randint(len(x), size=(len(x)))
    # set of unique sample identifiers in the bag
    unique = set(bag_indices)
    # select samples that are not in the bag
    out_of_bag_indices = [i for i in range(len(x)) if i not in unique]
    return x[bag_indices], y[bag_indices], x[out_of_bag_indices], y[out_of_bag_indices]


def fit(x, y, t, num_trees, max_tree_nodes, min_samples_leaf, min_samples_split, class_majority, measure, separate_max):
    forest = []
    for i in range(num_trees):
        x_train, y_train, x_out_of_bag, y_out_of_bag = bootstrap(x, y)

        tree = decision_tree.fit(
            x=x_train,
            y=y_train,
            t=t,
            randomized=True,
            max_tree_nodes=max_tree_nodes,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            class_majority=class_majority,
            measure=measure,
            separate_max=separate_max)
        forest.append(tree)
    return forest


def predict(forest, sample):
    predictions = [decision_tree.predict(tree, sample, dist=True) for tree in forest]
    y_dist = {k: v / float(len(forest)) for k, v in np.sum(predictions).iteritems()}
    return max(y_dist, key=y_dist.get)
