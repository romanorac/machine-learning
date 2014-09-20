import copy

def tree_view(tree, feature_names = [], stack=[0]):
	tree = copy.deepcopy(tree)
	
	while len(stack) > 0:
		while stack[0] not in tree:
			stack.pop(0)		
			if len(tree) == 0:
				return tree
			else:
				del(tree[stack[0]][0])

		node = tree[stack[0]][0]
		spaces = "".join(["| " for i in range(node[4])])
		name = "atr"+str(node[1]) if feature_names == [] else feature_names[node[1]]

		if node[1] == -1:
			name, operator,split = "root", "", ""
		elif node[5] == "c":
			operator = "<=" if len(tree[stack[0]]) == 2 else ">"
			split = node[2]
		else:
			operator = "in"
			split = sorted(node[2])
		
		print spaces, name , operator, split ,", Dist:", node[3], ", #Inst:", sum(node[3].values())

		new_stack = [k[0] for k in tree[stack[0]]] 
		if len(tree[stack[0]]) == 1:
			del(tree[stack[0]])
			stack.pop(0)
		stack = [new_stack[0]] + stack

	return tree










