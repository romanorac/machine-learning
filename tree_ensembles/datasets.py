import numpy as np

path = "/Users/hiphop/Documents/GitHub/IterativeML/tree_ensembles/datasets/"

def load_lymphography():
	name = "lymphography"
	dataset_type = "mixed features"

	t = ["d","d","d","d","d","d","d","d","c","c","d","d","d","d","d","d","d","c"]

	f = open(path + "lymphography.csv","r")
	data = np.loadtxt(f, delimiter = ",",dtype= np.object)
	x = np.array(data[:,range(1,len(t)+1)])
	y = data[:,0]

	for i in range(len(t)):
		if t[i] == "c":
			x[:, i] = np.array(x[:, i], dtype=np.float)

	feature_names = ["lymphatics","bl_affere","bl_lymph_c","bl_lymph_s","by_pass","extravasates","regen","early_uptake","lym_dimin","lym_enlar","changes_lym","defect","changes_node","changes_stru","spec_forms","dislocation","exclusion","no_nodes"]
	return x, y, t, feature_names, name, dataset_type
	

def load_breast_cancer():
	name = "breast_cancer"
	dataset_type = "discrete features"

	f = open(path + "breast_cancer_wisconsin_disc.csv","r")
	data = np.loadtxt(f, delimiter = ",",dtype= np.string0)
	x = np.array(data[:,range(1,10)],dtype =np.float)
	y = data[:,10]

	feature_names = ["Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion",   "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitose"]

	t = ["d","d","d","d","d","d","d","d","d"]
	return x, y, t, feature_names, name, dataset_type

def load_wine():
	name = "wine"
	dataset_type = "continuous features"

	f = open(path + "wine.csv","r")
	data = np.loadtxt(f, delimiter = ",",dtype= np.string0)
	x = np.array(data[:,range(1,14)], dtype = float)
	y = data[:,0]
	t = ["c" for i in range(13)]
	feature_names = ["a"+str(i) for i in range(1,14)]

	return x, y, t, feature_names, name, dataset_type

def load_bank():
	name = "bank"
	dataset_type = "mixed features"

	f = open(path + "bank.csv","r")
	data = np.loadtxt(f, delimiter = ";",dtype= np.object)
	t = ["c","d","d","d","d","c","d","d","d","c","d","c","c","c","c","d"]
	x = np.array(data[:,range(0,16)])
	y = data[:,16]

	feature_names = ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome"]

	
	for i in range(len(t)):
		if t[i] == "c":
			x[:,i] = np.array(x[:,i],dtype = float)

	return x, y, t, feature_names, name, dataset_type


def load_iris():
	name = "iris"
	dataset_type = "continuous features"

	f = open(path + "iris.csv","r")
	data = np.loadtxt(f, delimiter = ",",dtype= np.string0)
	x = np.array(data[:,range(0,4)],dtype =np.float)
	y = data[:,4]

	feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
	t = ["c","c","c","c"]
	return x, y, t, feature_names, name, dataset_type

def load_lenses():
	name = "lenses"
	dataset_type = "discrete features"

	f = open(path + "lenses.csv","r")
	data = np.loadtxt(f, delimiter = ",",dtype= np.string0)
	x = np.array(data[:,range(0,4)],dtype = np.string0)
	y = data[:,4]

	feature_names = ["age","prescription", "asigmatic","tear_rate"]
	t = ["d","d","d","d"]
	return x, y, t, feature_names, name, dataset_type

def load_lung():
	name = "lung cancer"
	dataset_type = "discrete features"

	f = open(path + "lung_cancer.csv","r")
	data = np.loadtxt(f, delimiter = ",",dtype= np.string0)
	x = np.array(data[:,range(1,57)],dtype = float)
	y = data[:,0]
	feature_names = ["a"+str(i) for i in range(1,57)]
	t = ["d" for i in range(1,57)]
	return x, y, t, feature_names, name, dataset_type

def load_car():
	name = "car"
	dataset_type = "discrete features"

	f = open(path + "car.csv","r")
	data = np.loadtxt(f, delimiter = ",",dtype= np.string0)
	x = np.array(data[:,range(0,6)],dtype = np.string0)
	y = data[:,6]

	feature_names = ["buying","maint","doors","persons","lugboot","safety"]
	t = ["d","d","d","d","d","d"]

	return x, y, t, feature_names, name, dataset_type

def load_segmentation():
	name = "segmentation dataset"
	dataset_type = "continuous features"

	f = open(path + "segmentation_combined.data","r")
	data = np.loadtxt(f, delimiter = ",",dtype= np.string0)

	x = np.array(data[:,range(1,20)],dtype = np.float)
	y = data[:,0]

	feature_names = ["REGION-CENTROID-COL","REGION-CENTROID-ROW","REGION-PIXEL-COUNT","SHORT-LINE-DENSITY-5","SHORT-LINE-DENSITY-2","VEDGE-MEAN","VEDGE-SD","HEDGE-MEAN","HEDGE-SD","INTENSITY-MEAN","RAWRED-MEAN","RAWBLUE-MEAN","RAWGREEN-MEAN","EXRED-MEAN","EXBLUE-MEAN","EXGREEN-MEAN","VALUE-MEAN","SATURATION-MEAN","HUE-MEAN"]
	t = ["c" for i in range(len(feature_names))]

	return x, y, t, feature_names, name, dataset_type
























