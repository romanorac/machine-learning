def tree_view(tree, feature_names, stack=[0]):
	import copy
	tree = copy.deepcopy(tree)
	operator = ""
	while len(stack) > 0:
		#if tree growing was stopped
		while stack[0] not in tree:
			stack.pop(0)		
			if len(tree) == 0:
				return tree
			else:
				del(tree[stack[0]][0])
		
		el = tree[stack[0]][0]
		spaces = "".join(["| " for i in range(el[4])])
		if isinstance(el[0], basestring):
			#leaf
			if el[5] == "c":
				operator = "<=" if len(tree[stack[0]]) == 2 else ">"
			name = "root" if el[1] == -1 else feature_names[el[1]]
			print spaces, name ,operator, el[2] if el[5] == "c" else sorted(el[2]), el[3], sum(el[3].values())
			del(tree[stack[0]][0])
			if len(tree[stack[0]]) == 0:
				del(tree[stack[0]])
				if len(stack) > 1:
					del(tree[stack[1]][0])
				stack.pop(0)

			continue
		#non leaf
		if el[5] == "c":
			operator = "<=" if len(tree[stack[0]]) == 2 else ">"
		name = "root" if el[1] == -1 else feature_names[el[1]]
		print spaces, name , operator, el[2] if el[5] == "c" else sorted(el[2]), el[3], sum(el[3].values())

		new_stack = [k[0] for k in tree[stack[0]] if isinstance(k[0], int)]
		#delete
		if len(tree[stack[0]]) == 1:
			del(tree[stack[0]])
			stack.pop(0)
		stack = [new_stack[0]] + stack

	return tree