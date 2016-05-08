import copy


def tree_view(tree, feature_names=[]):
    tree = copy.deepcopy(tree)
    output = ""
    stack = [0]

    if len(tree) > 0 and len(tree[0]) > 1:
        tree[0].pop(1)

    while len(stack) > 0:
        while stack[0] not in tree:
            stack.pop(0)
            if len(tree) == 0:
                return output
            else:
                del (tree[stack[0]][0])

        node = tree[stack[0]][0]
        spaces = "".join(["| " for i in range(node[4])])
        name = "atr" + str(node[1]) if feature_names == [] else feature_names[node[1]]

        if node[1] == -1:
            name, operator, split = "root", "", ""
        elif node[5] == "c":
            operator = " <= " if len(tree[stack[0]]) == 2 else " > "
            split = round(node[2], 4)
        else:
            operator = " in "
            str_values = sorted([str(value) for value in node[2]])
            split = "[" + ", ".join(str_values) + "]"

        output += spaces + name + operator + str(split) + ", Dist: " + str(node[3]) + ", #Inst: " + str(
            sum(node[3].values())) + "\n"

        new_stack = [k[0] for k in tree[stack[0]]]
        if len(tree[stack[0]]) == 1:
            del (tree[stack[0]])
            stack.pop(0)
        stack = [new_stack[0]] + stack

    return output
