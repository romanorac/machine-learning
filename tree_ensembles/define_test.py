import time
from sklearn import tree as sk_tree
import datasets
import random_forest
import decision_tree
import measures
import tree_view
import numpy as np
 
def nominal_feature_estimation_mdl():
	r_mdl_breastcancer = np.array([0.3560739,0.5810625,0.5610676,0.3692719,0.4871053,0.5130646,0.4832682,0.4504506, 0.1937647])
	x, y, t, feature_names, name, dataset_type = datasets.load_breast_cancer()
	estimation = np.array([measures.mdl(x[:,i], y, t[i])[0] for i in range(len(x[0]))])
	if np.allclose(estimation, r_mdl_breastcancer):
		print "nominal_feature_estimation_mdl", "succeeded"
	else:
		print "nominal_feature_estimation_mdl", "FAILED"

def nominal_feature_estimation_info_gain():
	r_infogain_breastcancer = np.array([0.3632432,0.5889188,0.5690254,0.3758817,0.4941868,0.5202378,0.4903110,0.4572384, 0.1993985])
	x, y, t, feature_names, name, dataset_type = datasets.load_breast_cancer()
	estimation = np.array([measures.info_gain(x[:,i], y, t[i])[0] for i in range(len(x[0]))])
	if np.allclose(estimation, r_infogain_breastcancer):
		print "nominal_feature_estimation_mdl", "succeeded"
	else:
		print "nominal_feature_estimation_mdl", "FAILED"

def gower_similarity_test():
	"""
	R CODE
	library(cluster)
	x1 <- c("brown", "blue", "red")
	x2 <- c("yellow","yellow","yellow")
	x3 <- c(1, 30, 20)
	x4 <- c(15, 12, 1)
	x <- data.frame(x1,x2,x3,x4)
	daisy(x, metric = "gower")
	"""
	types = ["d","d","c","c"]
	x1 = ["brown","yellow", 1, 15]
	x2 = ["blue","yellow", 30, 12]
	x3 = ["red","yellow", 20, 1]
	ranges = [29, 14]

	sim_x1_x2 = random_forest.gower_similarity(x1,x2,types,ranges)
	sim_x1_x3 = random_forest.gower_similarity(x1,x3,types,ranges)
	sim_x2_x3 = random_forest.gower_similarity(x2,x3,types,ranges)
	if np.allclose(sim_x1_x2, 0.5535714) and np.allclose(sim_x1_x3, 0.6637931) and np.allclose(sim_x2_x3, 0.5326355):
		print "gower_similarity_test", "succeeded"
	else:
		print "gower_similarity_test", "FAILED"

def decision_tree_ca(dataset, args):
	x, y, t, feature_names, name, dataset_type = dataset
	print "________________________________________"
	print name
	print dataset_type

	start = time.time()
	tree = decision_tree.fit(
		x = x, 
		y = y, 
		t = t, 
		max_tree_nodes = args["max_tree_nodes"], 
		randomized = False, 
		leaf_min_inst = args["leaf_min_inst"], 
		class_majority = args["class_majority"],  
		measure = args["measure"], 
		split_fun = args["split_fun"])
	end = time.time()

	if args["show_tree"] == True:
		tree2 = tree_view.tree_view(tree, feature_names, stack = [0])
	pred = [decision_tree.predict(tree,sample) for sample in x]
	ca = 1- (sum(pred != y)/float(len(y)))
	
	print "CA on training data", ca
	print "Elapsed time", round(end - start,2),"s"
	print

def decision_tree_measures(dataset, args):
	x, y, t, feature_names, name, dataset_type = dataset
	x_train, y_train, x_test, y_test = random_forest.bootstrap(x,y)
	print "________________________________________"
	print name, dataset_type
	
	start_ig = time.time()
	tree = decision_tree.fit(
		x = x_train, 
		y = y_train, 
		t = t, 
		max_tree_nodes = args["max_tree_nodes"], 
		randomized = False, 
		leaf_min_inst = args["leaf_min_inst"], 
		class_majority = args["class_majority"],  
		measure = measures.info_gain, 
		split_fun = args["split_fun"])
	end_ig = time.time()

	if args["show_tree"] == True:
		tree2 = tree_view.tree_view(tree, feature_names, stack = [0])

	pred_ig = [decision_tree.predict(tree, sample) for sample in x_test]
	ca_ig = 1- (sum(pred_ig != y_test)/float(len(y_test)))
	print "Classification accuracy"
	print "information gain", ca_ig
	
	start_mdl = time.time()
	tree = decision_tree.fit(
		x = x_train, 
		y = y_train, 
		t = t, 
		max_tree_nodes = args["max_tree_nodes"], 
		randomized = False, 
		leaf_min_inst = args["leaf_min_inst"], 
		class_majority = args["class_majority"],  
		measure = measures.mdl, split_fun = args["split_fun"])
	end_mdl = time.time()
	
	if args["show_tree"] == True:
		tree2 = tree_view.tree_view(tree, feature_names, stack = [0])
	pred_mdl = [decision_tree.predict(tree, sample) for sample in x_test]
	
	ca_mdl = 1- (sum(pred_mdl != y_test)/float(len(y_test)))
	print "mdl", ca_mdl
	print
	print "Elapsed time" 
	print "information gain", round(end_ig - start_ig,2),"s"
	print "mdl", round(end_mdl - start_mdl,2),"s"
	
def weighted_forest(dataset, args):
	x, y, t, feature_names, name, dataset_type = dataset
	x_train, y_train, x_test, y_test = random_forest.bootstrap(x,y)
	print "________________________________________"
	print name, dataset_type
	
	wforest, medoids, gower_range = random_forest.fit_weighted(
		x = x_train, 
		y = y_train, 
		t = t, 
		num_trees = args["num_trees"], 
		max_tree_nodes = args["max_tree_nodes"], 
		leaf_min_inst = args["leaf_min_inst"], 
		measure = args["measure"], 
		class_majority = args["class_majority"], 
		split_fun = args["split_fun"])
	y_pred3 = [random_forest.predict_weighted(wforest, medoids, gower_range, sample, t) for sample in x_test]
	wforest_ca = (sum(y_pred3 == y_test)/float(len(y_test)))
	print "Weighted forest CA", wforest_ca


def compare(dataset, args):
	x, y, t, feature_names, name, dataset_type = dataset
	x_train, y_train, x_test, y_test = random_forest.bootstrap(x,y)
	print "________________________________________"
	print name, dataset_type
	print
	
	#TREE
	start_tree = time.time()
	tree = decision_tree.fit(
		x = x_train, 
		y = y_train, 
		t = t, 
		max_tree_nodes = args["max_tree_nodes"], 
		leaf_min_inst = args["leaf_min_inst"], 
		class_majority = args["class_majority"],
		randomized = False,
		measure = args["measure"], 
		split_fun = args["split_fun"])
	end_tree = time.time()

	y_pred = [decision_tree.predict(tree, sample) for sample in x_test]
	tree_ca = 1- (sum(y_pred != y_test)/float(len(y_test)))
	print "Classification accuracy"
	print "Tree", tree_ca

	if dataset_type ==  "continuous features":
		#SKLEARN TREE
		clf = sk_tree.DecisionTreeClassifier(min_samples_leaf = args["leaf_min_inst"])
		clf = clf.fit(x_train, y_train)
		y_pred4 = clf.predict(x_test)
		print "Scikit tree", 1- (sum(y_pred4 != y_test)/float(len(y_test)))
	
	#RANDOM FOREST
	start_forest = time.time()
	forest = random_forest.fit(
		x = x_train, 
		y = y_train, 
		t = t, 
		num_trees = args["num_trees"],
		max_tree_nodes = args["max_tree_nodes"], 
		leaf_min_inst = args["leaf_min_inst"], 
		measure = args["measure"], 
		class_majority = args["class_majority"], 
		split_fun = args["split_fun"])
	end_forest = time.time()
	
	y_pred2 = [random_forest.predict(forest, sample) for sample in x_test]
	forest_ca = 1- (sum(y_pred2 != y_test)/float(len(y_test)))
	print "Forest", forest_ca
	
	
	#WEIGHTED RANDOM FOREST
	start_wforest = time.time()
	wforest, medoids, gower_range = random_forest.fit_weighted(
		x = x_train, 
		y = y_train, 
		t = t, 
		num_trees = args["num_trees"], 
		max_tree_nodes = args["max_tree_nodes"], 
		leaf_min_inst = args["leaf_min_inst"], 
		measure = args["measure"], 
		class_majority = args["class_majority"], 
		split_fun = args["split_fun"])
	end_wforest = time.time()
	
	y_pred3 = [random_forest.predict_weighted(wforest, medoids, gower_range, sample, t) for sample in x_test]
	wforest_ca = 1- (sum(y_pred3 != y_test)/float(len(y_test)))
	print "Weighted forest", wforest_ca
	
	
	print 
	print "Elapsed time"
	print "Tree", round(end_tree - start_tree,2),"s"
	print "Forest", round(end_forest - start_forest,2),"s"
	print "Weighted forest", round(end_wforest - start_wforest,2),"s"
	

































