import numpy as np
from collections import Counter
from itertools import combinations, permutations

import decision_tree
import k_medoids
import sys
import time

def bootstrap(x,y):	
	#generate len(x) random numbers with replacement from 0 to range(len(x))
	bag_indices = np.random.randint(len(x), size=(len(x)))
	#set of unique sample identifiers in the bag
	unique = set(bag_indices)
	#select samples that are not in the bag
	out_of_bag_indices = [i for i in range(len(x)) if i not in unique]

	return x[bag_indices], y[bag_indices], x[out_of_bag_indices], y[out_of_bag_indices]

def fit(x, y, t, num_trees, max_tree_nodes, leaf_min_inst, class_majority, measure, split_fun, intervals):
	"""
	Random forest algorithm
	"""

	forest = []
	for i in range(num_trees):
		x_train, y_train, x_out_of_bag, y_out_of_bag = bootstrap(x, y)
		
		tree = decision_tree.fit(
			x = x_train, 
			y = y_train, 
			t = t, 
			max_tree_nodes = max_tree_nodes, 
			leaf_min_inst = leaf_min_inst, 
			class_majority = class_majority, 
			randomized = True, 
			measure = measure, 
			split_fun = split_fun,
			intervals = intervals)
		forest.append(tree)	
	return forest

def predict(forest, sample):
	predictions = [decision_tree.predict(tree, sample) for tree in forest]
	y_dist = Counter(predictions)
	return max(y_dist, key=y_dist.get)

def gower_dissimilarity(x1, x2, types, ranges):
	"""
	Calculate gower dissimilarity

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
			gower += 1 if x1[i] == x2[i] else 0
	return 1 - gower/float(len(x1))

def fit_weighted(x, y, t, num_trees, max_tree_nodes, leaf_min_inst, class_majority, measure, split_fun, intervals):

	similarity_mat = {} #initialize similarity matrix
	forest, margins = [], []
	for i in range(num_trees):
		bag_indices = np.random.randint(len(x), size=(len(x)))
		unique = set(bag_indices)
		out_of_bag_indices = [i for i in range(len(x)) if i not in unique]

		tree = decision_tree.fit(
			x = x[bag_indices], #remove identifiers from decision tree
			y = y[bag_indices], 
			t = t, 
			max_tree_nodes = max_tree_nodes, 
			leaf_min_inst = leaf_min_inst, 
			class_majority = class_majority, 
			randomized = True, 
			measure = measure, 
			split_fun = split_fun,
			intervals = intervals)
		forest.append(tree)
		
		#calculate margins
		tree_margins, leafs_grouping = {}, {}
		for j in out_of_bag_indices:
			leaf, margin = decision_tree.predict(tree, x[j], y[j])
			tree_margins[j] = margin
			if leaf in leafs_grouping:
				leafs_grouping[leaf].append(j)
			else:
				leafs_grouping[leaf] = [j]	
		margins.append(tree_margins)
		
		for k, v in leafs_grouping.iteritems():
			for cx, cy in permutations(v,2): 
				if cx in similarity_mat:
					similarity_mat[cx][cy] = similarity_mat[cx].get(cy, 0) - 1
				else:
					similarity_mat[cx] = {cy: -1}

	min_elements = []
	for k, v in similarity_mat.iteritems():
		min_id = min(similarity_mat[k], key = similarity_mat[k].get) 
		min_elements.append((similarity_mat[k][min_id], min_id))
	min_elements = sorted(min_elements)

	k = int(np.sqrt(len(x[0]))) + 1 #sqrt(num_features)+1
	#k = len(x[0])
	#k = len(np.unique(y)) * len(np.unique(y))

	cidx = set()
	counter = 0
	while counter < len(min_elements) and len(cidx) < k:
		cidx.add(min_elements[counter][1])
		counter += 1
	inds, medoids_i = k_medoids.fit(similarity_mat,len(x), list(cidx))

	#for every cluster join its sample identifiers on separate list
	clusters = [np.where(inds == i)[0] for i in np.unique(inds)]
	medoids = x[medoids_i] #set medoids without sample identifiers

	stats = [[] for i in range(len(medoids_i))] 
	for i in range(len(forest)): #for every tree in forest
		for num, cluster in enumerate(clusters):
			#num - sequence of a cluster
			#cluster - all sample identifiers in cluster 

			#calculate average margin for cluster
			values = [margins[i][sample_id] for sample_id in cluster if int(sample_id) in margins[i]]
			if values != []:
				avg = np.average(values)
				forest[i]["margin" + str(num)] = avg
				stats[num].append(avg)
			
	stats = [np.median(value) for value in stats]
	gower_range = np.array([np.ptp(x[:,i]) for i in range(len(t)) if t[i] == "c"])
	gower_range[gower_range == 0] = 1e-9

	new_medoids = []
	for i in range(len(medoids)):
		cont, disc = [],[]
		for j in range(len(medoids[i])):
			if t[j] == "d":
				disc.append(medoids[i][j])
			else:
				cont.append(medoids[i][j])
		new_medoids.append((np.array(cont), np.array(disc)))
	
	return forest, new_medoids, stats, gower_range

def predict_weighted(forest, medoids, stats, gower_range, sample, t):
	"""
	Function uses weights for prediction of a sample
	"""

	#calculate similarity with sample and with every medoid
	#similarity = sorted([(round(gower_dissimilarity(sample, medoid, t, gower_range),4),i) for i, medoid in enumerate(medoids)])
	cont = np.array([sample[i] for i in range(len(sample)) if t[i] == "c"])
	disc = np.array([sample[i] for i in range(len(sample)) if t[i] == "d"])
	
	similarity = []
	#print gower_range
	for i, medoid in enumerate(medoids):
		gower = 1 - (sum(1 - np.true_divide(np.abs(cont - medoid[0]), gower_range)) + np.sum(disc == medoid[1]))/float(len(cont) + len(disc))
		similarity.append((round(gower,4), i))
	
	
	similarity = sorted(similarity)
	indices = [sim[1] for sim in similarity if similarity[0][0] == sim[0]]

	
 	global_predictions = {}
	for index in indices:

		predictions = {}
		margin = "margin"+str(index)
		for tree in forest:
			#make predictions with trees with selected margin
			if margin in tree: #and tree[margin] >= stats[index]:
				#for every predicted label, store its margins 
				pred = decision_tree.predict(tree, sample)
				predictions[pred] = predictions.get(pred, []) + [tree[margin]]


		for k, v in predictions.iteritems(): 
			predictions[k] = np.average(v) * len(v)
		#if predictions == {}:

		#else:

		max_pred = max(predictions, key = predictions.get)
		if max_pred not in global_predictions:
			global_predictions[max_pred] = predictions[max_pred]
		elif predictions[max_pred] > global_predictions[max_pred]:
			global_predictions[max_pred] = predictions[max_pred]

	return max(global_predictions, key = global_predictions.get) 
































