import numpy as np
from collections import Counter
from itertools import combinations
import decision_tree
import k_medoids

def bootstrap(x,y):	
	#generate len(x) random numbers with replacement from 0 to range(len(x))
	bag_indices = np.random.randint(len(x), size=(len(x)))
	#set of unique sample identifiers in the bag
	unique_bag = np.unique(bag_indices)
	#select samples that are not in the bag
	out_of_bag_indices = [i for i in range(len(x)) if i not in unique_bag]
	return x[bag_indices], y[bag_indices], x[out_of_bag_indices], y[out_of_bag_indices]

def fit(x, y, t, num_trees, max_tree_nodes, leaf_min_inst, class_majority, measure, split_fun):
	"""
	Random forest algorithm
	"""

	forest = []
	for i in range(num_trees):
		x_train, y_train, x_test, y_test = bootstrap(x, y)
		
		tree = decision_tree.fit(
			x = x_train, 
			y = y_train, 
			t = t, 
			max_tree_nodes = max_tree_nodes, 
			leaf_min_inst = leaf_min_inst, 
			class_majority = class_majority, 
			randomized = True, 
			measure = measure, 
			split_fun = split_fun)
		forest.append(tree)	
	return forest

def predict(forest, sample):
	predictions = [decision_tree.predict(tree, sample) for tree in forest]
	y_dist = Counter(predictions)
	return max(y_dist, key=y_dist.get)

def gower_similarity(x1, x2, types, ranges):
	"""
	Calculate gower similarity

	x1: numpy array - sample 1
	x2: numpy array - sample 2
	types: list of strings - feature types
	ranges: list of integers - range for every continuous feature
	"""
	gower = 0
	ranges_count = 0
	for i in range(len(x1)): #for every feature in sample
		if types[i] == "c": 
			#feature is continuous 
			gower += 1 - abs(x1[i] - x2[i])/(1 if ranges[ranges_count] == 0 else float(ranges[ranges_count]))
			ranges_count += 1
		else: 
			#feature is discrete
			gower+= 1 if x1[i] == x2[i] else 0
	return 1 - gower/float(len(x1))

def fit_weighted(x, y, t, num_trees, max_tree_nodes, leaf_min_inst, class_majority, measure, split_fun):
	"""
	Weighted random forest
	"""

	#Add identifiers to input dataset
	id_column = np.array([i for i in range(len(x))]).reshape(len(x),1) 
	x = np.concatenate((id_column,x ), axis = 1)
	
	#initialize similarity matrix
	similarity_mat = [[0 for i in range(len(x))] for j in range(len(x))]
	forest, margins, gower_range = [], [], []
	
	for i in range(num_trees):
		x_train, y_train, x_test, y_test = bootstrap(x, y)

		tree = decision_tree.fit(
			x = x_train[:,1:], #remove identifiers from decision tree
			y = y_train, 
			t = t, 
			max_tree_nodes = max_tree_nodes, 
			leaf_min_inst = leaf_min_inst, 
			class_majority = class_majority, 
			randomized = True, 
			measure = measure, 
			split_fun = split_fun,
			gower = True) #calculate ranges of continuous features
		
		#store gower ranges from every bootstrap sample for every continuous feature. 
		gower_range.append(tree["gower_range"]) 
		forest.append(tree)
		
		#calculate margins
		leafs, tree_margins = [], []
		for j in range(len(x_test)):
			#pass test sample without identifier
			leaf, margin = decision_tree.predict(tree, x_test[j,1:], y_test[j])
			leafs.append(leaf)
			tree_margins.append(margin)

		#join margins with test identifiers for each tree
		margins.append(dict(zip(x_test[:,0].tolist(), tree_margins))) 

		
		leafs_grouping = {}
		#For each leaf obtain its test identifier. Group test identifiers by leaf identifiers
		for j in range(len(leafs)):
			leafs_grouping[leafs[j]] = leafs_grouping.get(leafs[j], []) + [int(x_test[j,0])]

		for k, v in leafs_grouping.iteritems():
			#generate all combinations for every test identifier
			for cx, cy in list(combinations(v,2)): 
					#increase similarity twice, because of symmetric distance matrix
					similarity_mat[cx][cy] += 1 
					similarity_mat[cy][cx] += 1

	#Trees that are build from bootstrap samples does not necessarily see minimum and maximum of every continuous value. We calculate continuous ranges on the end of forest building process.
	gower_range_new = []
	for i in range(len(gower_range[0])):
		minimum = np.min([gower_range[j][i][0] for j in range(len(gower_range))])
		maximum = np.max([gower_range[j][i][1] for j in range(len(gower_range))])
		gower_range_new.append(abs(maximum - minimum))

	#k = len(np.unique(y))
	k = int(np.sqrt(len(x[0]))) + 1	#experimental - set number of medoids as sqrt(num_features)+1  
	#divide similarities with number of trees
	similarity_mat = np.true_divide(similarity_mat, num_trees) 
	inds, medoids_i = k_medoids.fit(similarity_mat, k) #calculate medoids 
	
	#for every cluster join its sample identifiers on separate list
	clusters = [np.where(inds == i)[0] for i in np.unique(inds)]

	medoids = x[medoids_i,1:] #set medoids without sample identifiers

	for i in range(len(forest)): #for every tree in forest
		for num, cluster in enumerate(clusters):
			#num - sequence of a cluster
			#cluster - all sample identifiers in cluster 

			#calculate average margin for cluster
			counter = 0
			suma = 0
			for sample_identifier in cluster:
				if sample_identifier in margins[i]: #if tree predicted sample_identifier
					suma += margins[i][sample_identifier]
					counter+=1
			if suma != 0:
				forest[i]["margin" + str(num)] = suma/float(counter) #average margin

	return forest, medoids, gower_range_new

def predict_weighted(forest, medoids, gower_range, sample, t):
	"""
	Function uses weights for prediction of a sample
	"""

	#calculate similarity with sample and with every medoid
	similarity = []
	for i, medoid in enumerate(medoids):
		similarity.append((gower_similarity(sample, medoid, t, gower_range), i))
	#select index of most similar medoid. Lowest value is most similar.
	comparison_i = sorted(similarity)[0][1] 

	predictions = {}
	for tree in forest:
		#make predictions with trees with selected margin
		if "margin"+str(comparison_i) in tree:
			#for every predicted label, store its margins 
			pred = decision_tree.predict(tree, sample)
			predictions[pred] = predictions.get(pred, []) + [tree["margin"+str(comparison_i)]]
	
	if predictions == {}: 
		#if there is no tree with selected margin. Make a prediction with all forest
		return predict(forest, sample)
	else:
		for k, v in predictions.iteritems(): 
			predictions[k] = np.average(v) #for every predicted label, calculate average margin
		
		#label with higest average margin is selected as prediction
		return max(predictions, key = predictions.get) 

































