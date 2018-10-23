""""""""""""""""""""
Cheryl Liao
CSC 578-701
Intro to Neural networks
""""""""""""""""""""""""""""""""'
import NN578_network as network
import numpy as np

#to print activation values:
********************************
net4 = network.load_network("iris-423.dat")
inst1 = (np.array([5.7, 3, 4.2, 1.2]), np.array([0., 1., 0.]))
x1 = np.reshape(inst1[0], (4, 1))
y1 = np.reshape(inst1[1], (3, 1))
sample1 = [(x1, y1)]
inst2 = (np.array([4.8, 3.4, 1.6, 0.2]), np.array([1., 0., 0.]))
x2 = np.reshape(inst2[0], (4, 1))
y2 = np.reshape(inst2[1], (3, 1))
sample2 = [(x2, y2)]
# Call SGD with one instance for training and another for testing
net4.SGD(sample1, 1, 1, 1.0, sample2)

#Xor dataset to determine early termination works:
******************************************************
ret = np.genfromtxt('../data/xor.csv', delimiter=',')
temp = np.array([(entry[:2],entry[2:]) for entry in ret])
temp_inputs = [np.reshape(x, (2, 1)) for x in temp[:,0]]
temp_results = [network.vectorize_target(2, y) for y in temp[:,1]] # conver
xor_data = list(zip(temp_inputs, temp_results))
net=network.Network([2,4,2])
net.SGD(xor_data,250,1,2.2,test_data=None)

# Load the bigger Iris data and experiment with parameters:
**************************************************************
def my_load_csv(fname, no_trainfeatures, no_testfeatures):
    ret = np.genfromtxt(fname, delimiter=',')
    data = np.array([(entry[:no_trainfeatures],entry[no_trainfeatures:]) for entry in ret])
    temp_inputs = [np.reshape(x, (no_trainfeatures, 1)) for x in data[:,0]] 
    temp_results = [np.reshape(y, (no_testfeatures, 1)) for y in data[:,1]] 
    dataset = list(zip(temp_inputs, temp_results))
    return dataset
iris_train = my_load_csv('../data/iris.csv', 4, 3)
net2 = network.load_network("iris-423.dat")

net2.SGD(iris_train, 1000, 10, 1, None)
