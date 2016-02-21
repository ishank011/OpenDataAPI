import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from math import exp

def solve(train_data, train_labels, test_data):
	x = fld(train_data, train_labels, test_data)
	y = svm(train_data, train_labels, test_data)
	z = randomForest(train_data, train_labels, test_data)
	print x, y, z
	return (x+y+z)/3

def fld(train_data, train_labels, test_data):

	N, d, M = train_data.shape[0], train_data.shape[1], 10
	count_0 = sum([1 if train_labels[i]==0 else 0 for i in range(N)])
	count_1 = sum([1 if train_labels[i]==1 else 0 for i in range(N)])
	mean_1 = np.array((sum(train_data[i] for i in range(N) if train_labels[i]==1))/count_1)
	mean_0 = np.array((sum(train_data[i] for i in range(N) if train_labels[i]==0))/count_0)
	#print  count_0, count_1, mean_0.shape, mean_1.shape

	sigma = np.cov(np.transpose(train_data))
	rank = np.linalg.matrix_rank(sigma)
	print "Rank: ", str(rank)
	k = rank/2

	h_x = 0
	for i in range(M):
		R = np.random.rand(k, d)
		temp1 = np.linalg.inv(np.dot(np.dot(R, sigma), np.transpose(R)))
		temp2 = test_data - (mean_0 + mean_1)/2.0
		temp3 = np.dot(np.dot(np.transpose(R), temp1), R)
		temp4 = mean_1 - mean_0
		h_x += np.dot(np.dot(temp4, temp3), np.transpose(temp2))

	h_x /= float(M)
	return 1/(1+exp(-1*h_x))

def randomForest(train_data, train_labels, test_data):
	model = RandomForestClassifier()
	model.fit(train_data, train_labels)
	return model.predict_proba(test_data)[0][0]

def svm(train_data, train_labels, test_data):
	model = SVC(probability=True)
	model.fit(train_data, train_labels)
	return model.predict_proba(test_data)[0][0]