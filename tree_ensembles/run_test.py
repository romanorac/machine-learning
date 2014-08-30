import datasets
import define_test
import measures


def set_random_seed(seed):
	import random
	import numpy as np
	random.seed(seed)
	np.random.seed(seed)
	print "random seed to",seed

def clasifier_comparison_test(test_cases):
	print 
	print "Comparison of DECISION TREE, RANDOM FOREST, WEIGHTED FOREST"
	args = {}
	args["num_trees"] = 20 
	args["max_tree_nodes"] = 20 
	args["leaf_min_inst"] = 5 
	args["class_majority"] = 1
	args["measure"] = measures.info_gain
	#args["measure"] = measures.mdl
	args["split_fun"] = measures.equal_freq_splits
	#args["split_fun"] = measures.random_splits
	print args
	
	for num_test in test_cases:
		if num_test == 1:
			define_test.compare(datasets.load_breast_cancer(), args)
			define_test.compare(datasets.load_car(), args)
			define_test.compare(datasets.load_lung(), args) #Decision tree does 100%
			define_test.compare(datasets.load_lenses(), args) #Too small
			
		elif num_test == 2:
			define_test.compare(datasets.load_segmentation(), args)
			define_test.compare(datasets.load_iris(), args)
			define_test.compare(datasets.load_wine(), args)
			
		elif num_test == 3:
			define_test.compare(datasets.load_bank(), args)
			define_test.compare(datasets.load_lymphography(), args)

def decision_tree_test(test_cases):
	print 
	print "TEST OF DECISION TREE ON TRAINING DATA"
	args = {}
	args["max_tree_nodes"] = 20 
	args["leaf_min_inst"] = 5  
	args["class_majority"] = 1
	args["show_tree"] = False
	args["measure"] = measures.info_gain
	#args["measure"] = measures.mdl
	args["split_fun"] = measures.equal_freq_splits
	#args["split_fun"] = measures.random_splits
	print args

	for num_test in test_cases:
		if num_test == 1:
			define_test.decision_tree_ca(datasets.load_breast_cancer(), args)
			define_test.decision_tree_ca(datasets.load_car(), args)
			define_test.decision_tree_ca(datasets.load_lung(), args)
			define_test.decision_tree_ca(datasets.load_lenses(), args)
		
		elif num_test == 2:
			define_test.decision_tree_ca(datasets.load_segmentation(), args)
			define_test.decision_tree_ca(datasets.load_iris(), args)
			define_test.decision_tree_ca(datasets.load_wine(), args)

		elif num_test == 3:
			define_test.decision_tree_ca(datasets.load_bank(), args)
			define_test.decision_tree_ca(datasets.load_lymphography(), args)

def decision_tree_measures_test(test_cases):
	print
	print "TEST OF DECISION TREE WITH IG & MDL"
	args = {}
	args["max_tree_nodes"] = 20 
	args["leaf_min_inst"] = 5 
	args["class_majority"] = 1
	args["show_tree"] = False
	args["split_fun"] = measures.equal_freq_splits
	#args["split_fun"] = measures.random_splits
	print args

	for num_test in test_cases:
		if num_test == 1:
			define_test.decision_tree_measures(datasets.load_breast_cancer(), args)
			define_test.decision_tree_measures(datasets.load_car(), args)
			define_test.decision_tree_measures(datasets.load_lung(), args)
			define_test.decision_tree_measures(datasets.load_lenses(), args)
		
		elif num_test == 2:
			define_test.decision_tree_measures(datasets.load_segmentation(), args)
			define_test.decision_tree_measures(datasets.load_iris(), args)
			define_test.decision_tree_measures(datasets.load_wine(), args)

		elif num_test == 3:
			define_test. decision_tree_measures(datasets.load_bank(), args)
			define_test. decision_tree_measures(datasets.load_lymphography(), args)

def weighted_forest_test(test_cases):
	print
	print "TEST OF WEIGHTED FOREST"
	args = {}
	args["num_trees"] = 20 
	args["max_tree_nodes"] = 20 
	args["leaf_min_inst"] = 5 
	args["class_majority"] = 0.98
	args["measure"] = measures.info_gain
	#args["measure"] = measures.mdl
	args["split_fun"] = measures.equal_freq_splits
	#args["split_fun"] = measures.random_splits
	print args
	
	for num_test in test_cases:
		if num_test == 1:
			define_test.weighted_forest(datasets.load_breast_cancer(), args)
			define_test.weighted_forest(datasets.load_car(), args)
			define_test.weighted_forest(datasets.load_lung(), args) #Decision tree does 100%
			define_test.weighted_forest(datasets.load_lenses(), args) #Too small
			
		elif num_test == 2:
			define_test.weighted_forest(datasets.load_segmentation(), args)
			define_test.weighted_forest(datasets.load_iris(), args)
			define_test.weighted_forest(datasets.load_wine(), args)
			
		elif num_test == 3:
			define_test.weighted_forest(datasets.load_bank(), args)
			define_test.weighted_forest(datasets.load_lymphography(), args)

def gower_similarity_test():
	define_test.gower_similarity_test()

def nominal_feature_estimation_mdl():
	define_test.nominal_feature_estimation_mdl()

def nominal_feature_estimation_info_gain():
	define_test.nominal_feature_estimation_info_gain()

if __name__ == '__main__':
	set_random_seed(5)
	gower_similarity_test()
	nominal_feature_estimation_mdl()
	nominal_feature_estimation_info_gain()
	print
	clasifier_comparison_test([1,2,3])
	decision_tree_test([1,2,3])
	decision_tree_measures_test([1,2,3])
	weighted_forest_test([1,2,3])
	

	

















