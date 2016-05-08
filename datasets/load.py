from os import path

import numpy as np

import datasets

datasets_root = path.join(datasets.__path__[0], "data", "")


def lymphography():
    name = "lymphography"
    dataset_type = "mixed features"

    t = ["d", "d", "d", "d", "d", "d", "d", "d", "c", "c", "d", "d", "d", "d",
         "d", "d", "d", "c"]

    data = np.loadtxt(path.join(datasets_root, "lymphography.csv"), delimiter=",", dtype=np.object)
    x = np.array(data[:, range(1, len(t) + 1)])
    y = data[:, 0]

    for i in range(len(t)):
        if t[i] == "c":
            x[:, i] = np.array(x[:, i], dtype=np.float)

    feature_names = ["lymphatics", "bl_affere", "bl_lymph_c", "bl_lymph_s",
                     "by_pass", "extravasates", "regen", "early_uptake",
                     "lym_dimin", "lym_enlar", "changes_lym", "defect",
                     "changes_node", "changes_stru", "spec_forms",
                     "dislocation", "exclusion", "no_nodes"]
    return x, y, t, feature_names, name, dataset_type


def breast_cancer():
    name = "breast_cancer"
    dataset_type = "discrete features"

    data = np.loadtxt(path.join(datasets_root, "breast_cancer_wisconsin_disc.csv"), delimiter=",", dtype=np.string0)
    x = np.array(data[:, range(1, 10)], dtype=np.float)
    y = data[:, 10]

    feature_names = ["Clump Thickness", "Uniformity of Cell Size",
                     "Uniformity of Cell Shape", "Marginal Adhesion",
                     "Single Epithelial Cell Size", "Bare Nuclei",
                     "Bland Chromatin", "Normal Nucleoli", "Mitose"]

    t = ["d", "d", "d", "d", "d", "d", "d", "d", "d"]
    return x, y, t, feature_names, name, dataset_type


def wine():
    name = "wine"
    dataset_type = "continuous features"

    data = np.loadtxt(path.join(datasets_root, "wine.csv"), delimiter=",", dtype=np.string0)
    x = np.array(data[:, range(1, 14)], dtype=float)
    y = data[:, 0]
    t = ["c" for i in range(13)]
    feature_names = ["a" + str(i) for i in range(1, 14)]

    return x, y, t, feature_names, name, dataset_type


def bank():
    name = "bank"
    dataset_type = "mixed features"

    data = np.loadtxt(path.join(datasets_root, "bank.csv"), delimiter=";", dtype=np.object)
    t = ["c", "d", "d", "d", "d", "c", "d", "d", "d", "c", "d", "c", "c", "c",
         "c", "d"]
    x = np.array(data[:, range(0, 16)])
    y = data[:, 16]

    feature_names = ["age", "job", "marital", "education", "default",
                     "balance", "housing", "loan", "contact", "day", "month",
                     "duration", "campaign", "pdays", "previous", "poutcome"]

    for i in range(len(t)):
        if t[i] == "c":
            x[:, i] = np.array(x[:, i], dtype=float)

    return x, y, t, feature_names, name, dataset_type


def iris():
    name = "iris"
    dataset_type = "continuous features"

    data = np.loadtxt(path.join(datasets_root, "iris.csv"), delimiter=",", dtype=np.string0)
    x = np.array(data[:, range(0, 4)], dtype=np.float)
    y = data[:, 4]

    feature_names = ["sepal_length", "sepal_width", "petal_length",
                     "petal_width"]
    t = ["c", "c", "c", "c"]
    return x, y, t, feature_names, name, dataset_type


def car():
    name = "car"
    dataset_type = "discrete features"

    data = np.loadtxt(path.join(datasets_root, "car.csv"), delimiter=",", dtype=np.string0)
    x = np.array(data[:, range(0, 6)], dtype=np.string0)
    y = data[:, 6]

    feature_names = ["buying", "maint", "doors", "persons", "lugboot",
                     "safety"]
    t = ["d", "d", "d", "d", "d", "d"]

    return x, y, t, feature_names, name, dataset_type


def segmentation():
    name = "segmentation dataset"
    dataset_type = "continuous features"

    data = np.loadtxt(path.join(datasets_root, "segmentation_combined.csv"), delimiter=",", dtype=np.string0)

    x = np.array(data[:, range(1, 20)], dtype=np.float)
    y = data[:, 0]

    feature_names = ["REGION-CENTROID-COL", "REGION-CENTROID-ROW",
                     "REGION-PIXEL-COUNT", "SHORT-LINE-DENSITY-5",
                     "SHORT-LINE-DENSITY-2", "VEDGE-MEAN", "VEDGE-SD",
                     "HEDGE-MEAN", "HEDGE-SD", "INTENSITY-MEAN", "RAWRED-MEAN",
                     "RAWBLUE-MEAN", "RAWGREEN-MEAN", "EXRED-MEAN",
                     "EXBLUE-MEAN", "EXGREEN-MEAN", "VALUE-MEAN",
                     "SATURATION-MEAN", "HUE-MEAN"]
    t = ["c" for i in range(len(feature_names))]

    return x, y, t, feature_names, name, dataset_type


def lwlr():
    name = "lwlr"
    dataset_type = "continuous features"
    t = ["c", "c"]

    data = np.loadtxt(path.join(datasets_root, "lwlr.csv"), delimiter=",")
    # add a column of ones to samples
    samples = np.insert(data[:, 0].reshape(len(data), 1), 0, np.ones(len(data)), axis=1)
    targets = data[:, 1]
    return samples, targets, t, name, dataset_type


def ex2():
    name = "ex2"
    dataset_type = "continuous features"
    t = ["c", "c"]
    data = np.loadtxt(path.join(datasets_root, "ex2.csv"), delimiter=",")
    samples, targets = data[:, 0], data[:, 1]
    # add a column of ones to samples
    samples = np.insert(samples.reshape(len(samples), 1), 0, np.ones(len(samples)), axis=1)
    return samples, targets, t, name, dataset_type
