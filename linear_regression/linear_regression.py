"""
Linear regression

Description of the algorithm can be found on 
http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex2/ex2.html
"""
import numpy as np
import matplotlib.pyplot as pyplot

#read the data
f = open("ex2.dat","r")
data = np.loadtxt(f, delimiter = ",")
x, y  = data[:,0], data[:,1] 
pyplot.scatter(x,y) #plot the data

#The fit phase of the linear regression
x = np.insert(x.reshape(len(x),1), 0, np.ones(len(x)), axis = 1) #add ones to x
thetas = np.linalg.lstsq(np.dot(x.transpose(),x), np.dot(x.transpose(),y))[0] #A^(-1) * b 
print "thetas",thetas

#plot a line
line_y = [x_el[1] * thetas[1] + thetas[0] for x_el in x] #y = kx + n 
pyplot.plot(x[:,1],line_y)
pyplot.show()


