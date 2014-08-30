"""
Decision tree algorithm

Algorithm builds a binary decision tree. It expands nodes in priority order, where priority is set by measure (information gain or minimum description length).

"""

import numpy as np
from collections import Counter
import Queue
import random

def rand_indices(x, rand_attr):
	"""
	Function randomly selects features without replacement. It used with random forest. Selected features must have more than one distinct value.
	x: numpy array - dataset
	rand_attr - parameter defines number of randomly selected features
	"""
	loop = True
	indices = range(len(x[0]))

	while loop:
		loop = False
		#randomly selected features without replacement
		rand_list = random.sample(indices, rand_attr) 
		for i in rand_list:
			if len(np.unique(x[:,i])) == 1:
				loop = True
				indices.remove(i)
				if len(indices) == rand_attr-1:
					return -1 #all features in dataset have one distinct value
				break
	return rand_list

def fit(x, y, t, randomized, max_tree_nodes, leaf_min_inst, class_majority, measure, split_fun, gower = False):
	"""
	Function builds a binary decision tree with given dataset and it expand tree nodes in priority order. Tree model is stored in a dictionary and it has following structure: 
	{parent_identifier: [(child_identifier, highest_estimated_feature_index , split_value, distribution_of_labels, depth, feature_type)]}

	x: numpy array - dataset with features
	y: numpy array - dataset with labels
	t: list - features types
	randomized: boolean - if True, algorithm estimates sqrt(num_of_features)+1 randomly selected features each iteration. If False, it estimates all features in each iteration.
	max_tree_nodes: integer - number of tree nodes to expand. 
	leaf_min_inst: float - minimal number of samples in leafs.
	class_majority: float - purity of the classes in leafs.
	measure: measure function - information gain or mdl.
	split_fun: split function - discretization of continuous features can be made randomly or with equal label frequency.
	gower: boolean - ranges of continuous features are calculated. They are needed for calculation of Gower similarity. 
	"""

	depth = 0 #depth of the tree
	node_id = 1 #node identifier 

	#conditions of continuous and discrete features. 
	operation = {"c": (np.less_equal, np.greater), "d": (np.in1d, np.in1d)}
	tree = {0:[(node_id, -1 ,"", dict(Counter(y)), depth, "")]} #initialize tree model
	
	if gower:
		#minimum and maximum value are stored for every continuous feature 
		tree["gower_range"] = [[np.min(x[:,i]),np.max(x[:,i])] for i in range(len(t)) if t[i] == "c"]
		
	mapping = range(len(x[0])) #global features indices
	rand_attr = int(np.sqrt(len(x[0]))) + 1	#sqrt(num_of_features)+1 is estimated if randomized == True. If randomized == False, all indices are estimated at each node.
	est_indices = rand_indices(x, rand_attr) if randomized else range(len(x[0])) 
	
	#estimate indices with given measure
	est = [measure(x[:,i], y, t[i], split_fun = split_fun) for i in est_indices]
	max_est, split = max(est) #find highest estimated split
	best = est_indices[est.index((max_est, split))] #select feature index with highest estimate
	
	queue = Queue.PriorityQueue() #initialize priority queue
	#put datasets in the queue
	queue.put((max_est, (node_id, x,y, mapping, best, split, depth)))

	while not queue.empty() and len(tree)*2 < max_tree_nodes: 
		_, (parent_id, x, y, mapping, best, split, depth) = queue.get()
		
		#features indices are mapped due to constantly changing subsets of data 
		best_map = mapping[best] 
		for j in range(2): #for left and right branch of the tree
			selection = range(len(x[0])) #select all indices for the new subset
			new_mapping = [i for i in mapping] #create a mapping of indices for a new subset
			
			if t[best_map] == "d" and len(split[j]) == 1:
				#if feature is discrete with one value in split, we cannot split it further. 
				selection.remove(best) #remove feature from new dataset
				new_mapping.remove(best_map) #remove mapping of feature

			#select rows of new dataset that satisfy condition (less than, greater than or in)
			indices = operation[t[best_map]][j](x[:,best],split[j]).nonzero()[0]
			#create new subsets of data
			sub_x, sub_y = x[indices.reshape(len(indices),1), selection], y[indices]
			
			node_id += 1 #increase node identifier
			y_dist = Counter(sub_y) #distribution of labels in the new node

			#connect child node with its parent and update tree model
			tree[parent_id] = tree.get(parent_id, []) + [(node_id, best_map, split[j], dict(y_dist), depth+1, t[best_map])]

			#select new indices for estimation
			est_indices = rand_indices(sub_x, rand_attr) if randomized and len(sub_x[0]) > rand_attr else range(len(sub_x[0]))
			#check label majority
			curent_majority = y_dist[max(y_dist, key = y_dist.get)]/float(len(sub_y))

			#if new node satisfies following conditions it can be further split 
			if curent_majority < class_majority and len(sub_y) > leaf_min_inst and est_indices != -1: 
				#estimate selected indices
				est = [measure(sub_x[:,i], sub_y, t[new_mapping[i]],split_fun=split_fun) for i in est_indices]
				max_est, new_split = max(est) #find highest estimated split
				#select feature index with highest estimate
				new_best = est_indices[est.index((max_est, new_split))]

				#put new datasets in the queue with inverse value of estimate (priority order)
				queue.put((max_est*-1,(node_id, sub_x, sub_y, new_mapping, new_best, new_split, depth+1)))
	return tree

def predict(tree, x, y = []):
	"""
	Function makes a prediction of one sample with a tree model. If y label is defined it returns node identifier and margin.

	tree: dictionary - tree model
	x: numpy array - one sample from the dataset
	y: string, integer or float - sample label
	"""

	#conditions of continuous and discrete features
	operation = {"c": (np.less_equal, np.greater), "d": (np.in1d, np.in1d)}
	node_id = 1 #initialize node identifier as first node under the root
	make_prediction = False #prediction is made when tree cannot be traversed further
	
	while 1:
		feature_index = tree[node_id][0][1] #select highest estimated feature index 
		
		no_children = True #if sample value is not in left or right branch
		for i in range(2): #for left and right branch
			node = tree[node_id][i] #set node 
			#check if sample value satisfies the condition
			if operation[tree[node_id][0][5]][i](x[feature_index], node[2]):
				no_children = False #node has children
				node_id = node[0] #set identifier of child node
				if node_id not in tree.keys(): #check if tree can be traversed further
					#node is a leaf
					y_dist = node[3] #save distribution of labels
					make_prediction = True
				break

		if no_children: 
			#value is not in left or right branch. This is only possible with discrete features
			#get label distributions of left and right child
			a, b = tree[node_id][0][3], tree[node_id][1][3] 
			#sum labels distribution to get parent label distribution
			y_dist = { k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b) }
			make_prediction = True
		
		if make_prediction:
			#calculate and sort probabilities of label distribution in current node
			probs = sorted(zip(np.true_divide(y_dist.values(), np.sum(y_dist.values())), y_dist.keys()), reverse = True)
			y_prob, prediction = probs[0] #label with highest probability
			
			if y == []: #if y is not defined, return prediction
				return prediction

			#calculate margin
			elif prediction == y: #if predicted label is correct
				#margin = predicted_label_prob - second_label_prob
				#if there is just one label to predict return its probability
				margin = y_prob - probs[1][0] if len(probs) > 1 else y_prob
			
			else: #if predicted label is incorrect
				probs = dict((pred, prob) for prob, pred in probs) #map probabilities to dictionary
				#margin = correct_label_prob - predicted_label_prob
				#if correct label was not in label distribution, it gets 0 probability
				margin = probs.get(y, 0) - y_prob
			return node_id, margin 





