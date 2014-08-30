"""
Special purpose k - medoids algorithm 
"""

import numpy as np
import random

def fit(D, k):
	"""
	Algorithm maximizes energy between clusters, which is distinction in this algorithm. Distance matrix contains mostly 0, which are overlooked due to search of maximal distances. Algorithm does not try to retain k clusters. 

	D: numpy array - Symmetric distance matrix
	k: int - number of clusters
	"""
	n = len(D)
	global_highest_energy = 0 #try to maximize global energy
	
	for i in range(10): #restart algorithm 10 times, because of randomness
		
		#select k random samples from distance matrix without replacement
		cidx = random.sample(range(n), k) 
		local_highest_energy = 0
		
		count = 0 #counter of iterations
		while 1:

			#select indices in each sample that maximizes its dimension
			inds = np.argmax(D[cidx],axis = 0) 
			
			#update centers
			cidx = []
			energy = 0 #current enengy
			for i in range(k):
				indsi = np.where(inds == i)[0] #find indices for every cluster
				if indsi != []: #if there is no indices in current cluster, remove it
					suma = np.sum(D[indsi][:,indsi], axis = 1)
					minind = np.argmax(suma) 
					energy += suma[minind] #increase energy
					cidx.append(indsi[minind]) #new centers
			
			if local_highest_energy == energy or count > 20:
				break
			if energy > local_highest_energy:
				local_highest_energy = energy
				if local_highest_energy > global_highest_energy:
					global_highest_energy = local_highest_energy
					inds_highest = inds
					cidx_highest = cidx
			count+=1
	
	#print "highest_energy", global_highest_energy
	
	return inds_highest, cidx_highest #cluster for every instance, medoids indices















