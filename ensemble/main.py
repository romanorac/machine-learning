import random
import time

import numpy as np
from sklearn import tree as sk_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import ensemble.decision_tree
import ensemble.measures
import ensemble.measures
import ensemble.model_view
import ensemble.random_forest
from datasets import load


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def decision_tree_ca(dataset, args):
    x, y, t, feature_names, name, dataset_type = dataset
    print "________________________________________"
    print name, dataset_type
    set_random_seed(args["seed"])
    x_train, y_train, x_test, y_test = ensemble.random_forest.bootstrap(x, y)
    start = time.time()
    tree = ensemble.decision_tree.fit(
        x=x_train,
        y=y_train,
        t=t,
        max_tree_nodes=args["max_tree_nodes"],
        randomized=False,
        min_samples_leaf=args["min_samples_leaf"],
        min_samples_split=args["min_samples_split"],
        class_majority=args["class_majority"],
        measure=args["measure"],
        separate_max=args["separate_max"])
    end = time.time()

    if args["show_tree"]:
        tree_model = ensemble.model_view.tree_view(tree, feature_names)
        print tree_model

    pred = [ensemble.decision_tree.predict(tree, sample) for sample in x_test]
    ca = 1 - (sum(y_test != pred) / float(len(pred)))

    print "DT CA", ca
    if args["show_time"]:
        print "Elapsed time", round(end - start, 2), "s"
        print


def compare(dataset, args):
    x, y, t, feature_names, name, dataset_type = dataset
    print "________________________________________"
    print name, dataset_type
    print

    set_random_seed(args["seed"])
    x_train, y_train, x_test, y_test = ensemble.random_forest.bootstrap(x, y)

    # TREE
    start_tree = time.time()
    tree = ensemble.decision_tree.fit(
        x=x_train,
        y=y_train,
        t=t,
        max_tree_nodes=args["max_tree_nodes"],
        randomized=False,
        min_samples_leaf=args["min_samples_leaf"],
        min_samples_split=args["min_samples_split"],
        class_majority=args["class_majority"],
        measure=args["measure"],
        separate_max=args["separate_max"])
    end_tree = time.time()

    y_pred = [ensemble.decision_tree.predict(tree, sample) for sample in x_test]
    tree_ca = 1 - (sum(y_pred != y_test) / float(len(y_test)))
    print "Classification accuracy"
    print "DT", tree_ca

    # RANDOM FOREST
    set_random_seed(args["seed"])
    start_forest = time.time()
    forest = ensemble.random_forest.fit(
        x=x_train,
        y=y_train,
        t=t,
        num_trees=args["num_trees"],
        max_tree_nodes=args["max_tree_nodes"],
        min_samples_leaf=args["min_samples_leaf"],
        min_samples_split=args["min_samples_split"],
        class_majority=args["class_majority"],
        measure=args["measure"],
        separate_max=args["separate_max"])
    end_forest = time.time()

    y_pred2 = [ensemble.random_forest.predict(forest, sample) for sample in x_test]
    forest_ca = 1 - (sum(y_pred2 != y_test) / float(len(y_test)))
    print "RF", forest_ca

    # SKLEARN DT and RF
    set_random_seed(args["seed"])
    dt_clf = sk_tree.DecisionTreeClassifier(min_samples_leaf=args["min_samples_leaf"],
                                            min_samples_split=args["min_samples_split"])
    rf_clf = RandomForestClassifier(n_estimators=args["num_trees"],
                                    criterion='entropy',
                                    max_depth=args["max_tree_nodes"],
                                    min_samples_leaf=args["min_samples_leaf"],
                                    min_samples_split=args["min_samples_split"],
                                    max_features='auto',
                                    bootstrap=True,
                                    oob_score=False,
                                    n_jobs=2,
                                    random_state=args["seed"],
                                    verbose=0)

    if dataset_type != "continuous features":
        le = LabelEncoder()
        for i in range(len(x_train[0])):
            x_train[:, i] = le.fit_transform((x_train[:, i]))
        for i in range(len(x_test[0])):
            x_test[:, i] = le.fit_transform((x_test[:, i]))
    dt_clf = dt_clf.fit(x_train, y_train)
    dt_pred = dt_clf.predict(x_test)
    rf_clf = rf_clf.fit(x_train, y_train)
    rf_pred = rf_clf.predict(x_test)

    print "Scikit DT", 1 - (sum(dt_pred != y_test) / float(len(y_test)))
    print "Scikit RF", 1 - (sum(rf_pred != y_test) / float(len(y_test)))

    if args["show_time"]:
        print
        print "Elapsed time"
        print "Tree", round(end_tree - start_tree, 2), "s"
        print "Forest", round(end_forest - start_forest, 2), "s"


def clasifier_comparison(test_cases, args):
    print "Comparison of DECISION TREE, RANDOM FOREST"
    print args

    for num_test in test_cases:
        if num_test == 1:
            compare(load.breast_cancer(), args)
            compare(load.car(), args)

        elif num_test == 2:
            compare(load.segmentation(), args)
            compare(load.iris(), args)
            compare(load.wine(), args)

        if num_test == 3:
            compare(load.bank(), args)
            compare(load.lymphography(), args)


def dt_multiple_datasets(test_cases, args):
    print "DECISION TREE"
    print args

    for num_test in test_cases:
        if num_test == 1:
            decision_tree_ca(load.breast_cancer(), args)
            decision_tree_ca(load.car(), args)

        elif num_test == 2:
            decision_tree_ca(load.segmentation(), args)
            decision_tree_ca(load.iris(), args)
            decision_tree_ca(load.wine(), args)

        elif num_test == 3:
            decision_tree_ca(load.bank(), args)
            decision_tree_ca(load.lymphography(), args)


if __name__ == '__main__':
    args = {}
    args["num_trees"] = 50
    args["max_tree_nodes"] = 50
    args["min_samples_leaf"] = 2
    args["min_samples_split"] = 5
    args["class_majority"] = 1
    args["measure"] = ensemble.measures.info_gain
    args["separate_max"] = True

    args["show_time"] = False
    args["show_tree"] = True
    args["seed"] = 1

    dt_multiple_datasets([1, 2, 3], args)
    clasifier_comparison([1, 2, 3], args)
