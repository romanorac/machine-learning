import datasets
import define_test
import measures

def clasifier_comparison_test(test_cases, args):
	print "Comparison of DECISION TREE, RANDOM FOREST, WEIGHTED FOREST"
	print args
	
	for num_test in test_cases:
		if num_test == 1:
			define_test.compare(datasets.load_breast_cancer(), args)
			define_test.compare(datasets.load_car(), args)
			#define_test.compare(datasets.load_lung(), args) #Decision tree does 100%
			#define_test.compare(datasets.load_lenses(), args) #Too small
			
		elif num_test == 2:
			define_test.compare(datasets.load_segmentation(), args)
			define_test.compare(datasets.load_iris(), args)
			define_test.compare(datasets.load_wine(), args)
			
		elif num_test == 3:
			define_test.compare(datasets.load_bank(), args)
			define_test.compare(datasets.load_lymphography(), args)
		elif num_test == 4:
			define_test.compare(datasets.load_segmentation_big(), args)

def decision_tree_test(test_cases, args):
	print "DECISION TREE"
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

def decision_tree_measures_test(test_cases, args):
	print "DECISION TREE WITH IG & MDL"
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

def weighted_forest_test(test_cases, args):
	print "TEST OF WEIGHTED FOREST"
	print args
	print
	for num_test in test_cases:
		if num_test == 1:
			define_test.weighted_forest(datasets.load_breast_cancer(), args)
			define_test.weighted_forest(datasets.load_car(), args)
			#define_test.weighted_forest(datasets.load_lung(), args) #Decision tree does 100%
			#define_test.weighted_forest(datasets.load_lenses(), args) #Too small
			
		elif num_test == 2:
			define_test.weighted_forest(datasets.load_segmentation(), args)
			define_test.weighted_forest(datasets.load_iris(), args)
			define_test.weighted_forest(datasets.load_wine(), args)
			
		elif num_test == 3:
			#define_test.weighted_forest(datasets.load_bank(), args)
			define_test.weighted_forest(datasets.load_lymphography(), args)

def gower_dissimilarity_test():
	define_test.gower_dissimilarity_test()

def nominal_feature_estimation_mdl():
	define_test.nominal_feature_estimation_mdl()

def nominal_feature_estimation_info_gain():
	define_test.nominal_feature_estimation_info_gain()

def margin_test():
	y_dist = {"a":0.6, "b":0.3, "c":0.1}
	correct_margins = [0.3, -0.5]
	correct_y = ["a", "c"]
	define_test.margin_test(y_dist, correct_y, correct_margins)

	y_dist = {"a":1}
	correct_margins = [1,-1]
	correct_y = ["a","b"]
	define_test.margin_test(y_dist, correct_y, correct_margins)

if __name__ == '__main__':
	args = {}
	args["num_trees"] = 50 
	args["max_tree_nodes"] = 50
	args["leaf_min_inst"] = 5 
	args["class_majority"] = 1
	args["measure"] = measures.info_gain
	#args["measure"] = measures.mdl
	args["split_fun"] = measures.equal_freq_splits
	#args["split_fun"] = measures.random_splits
	args["intervals"] = 10
		
	args["show_time"] = False
	args["show_tree"] = False
	args["seed"] = 1

	margin_test()
	gower_dissimilarity_test()
	nominal_feature_estimation_mdl()
	nominal_feature_estimation_info_gain()
	print
	decision_tree_test([1,2,3], args)
	decision_tree_measures_test([1,2,3], args)
	clasifier_comparison_test([1,2,3], args)


	

















