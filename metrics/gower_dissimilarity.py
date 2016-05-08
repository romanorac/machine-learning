def gower_dissimilarity(x1, x2, types, ranges):
    """
    Calculate gower dissimilarity

    :param x1: numpy array - sample 1
    :param x2: numpy array - sample 2
    :param types: list of strings - feature types
    :param ranges: list of integers - range for every continuous feature

    :return gower dissimilarity
    """
    gower = 0
    ranges_count = 0
    for i in range(len(x1)):
        if types[i] == "c":
            # feature is continuous
            gower += 1 - abs(x1[i] - x2[i]) / (1 if ranges[ranges_count] == 0 else float(ranges[ranges_count]))
            ranges_count += 1
        else:
            # feature is discrete
            gower += 1 if x1[i] == x2[i] else 0
    return 1 - gower / float(len(x1))
