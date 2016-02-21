from django.shortcuts import render, render_to_response, get_object_or_404, redirect
from django.http import HttpResponseRedirect, HttpResponse, JsonResponse
import numpy as np
import ml_classifiers
import csv
from sklearn.preprocessing import Imputer


def diabetes_pregnancy(request):
	
	f1 = open('static/data1/pima-indians-diabetes.data', 'r')		#Data is stored in the given file in static folder
	data = [list(map(float, a.split(','))) for a in f1.readlines()]
	train_labels = [int(a.pop()) for a in data]
	train_data = np.array(data)
	train_mean = np.mean(train_data, axis=0)
	f1.close()
	del data
	#print train_mean
	#print train_data.shape, len(train_labels)

	params = ['times_pregnant', 'glucose_tol', 'diastolic_pb', 'triceps', 'insulin', 'mass_index', 'pedigree', 'age']	#The 8 parameters to be entered by user
	test_data, c_test, inputs, error = np.array([-1 for i in range(len(params))]), 0, {}, ''
	for i in range(len(params)):
		if params[i] in request.GET and request.GET[params[i]]:
			try:
				test_data[i] = float(request.GET[params[i]])
				inputs[params[i]] = test_data[i]
			except ValueError:
				error = 'ValueError. Please enter only integer or floating point values'
				inputs[params[i]] = request.GET[params[i]]
			c_test += 1
		else:
			test_data[i] = train_mean[i]
			inputs[params[i]] = '-'

	if c_test<4:
		error = 'Not enough parameters (at least 4 required)'	#Requires at least 4 out of 8 parameters

	if error:
		return JsonResponse({'Inputs given':inputs, 'Error':error})

	risk = ml_classifiers.solve(train_data, train_labels, test_data)	#Takes the average prediction of 3 state of the art classifiers
	if risk > 0.5:
		pred = 1
	else:
		pred = 0
	f1 = open('static/data1/pima-indians-diabetes.data', 'a')		#Writes the test data to the file as a new training sample
	f1.write(','.join([str(a) for a in test_data])+','+str(pred)+'\n')
	f1.close()
	json_data = {'Inputs given':inputs, 'Risk factor':risk}
	return JsonResponse(json_data)


def mortality(request):

	f1 = open('static/data2/mortality.data', 'r')		#Data is stored in the given file in static folder
	data = [list(map(float, a.split(','))) for a in f1.readlines()]
	train_labels = [int(a.pop()) for a in data]
	train_data = np.array(data)
	train_mean = np.mean(train_data, axis=0)
	f1.close()
	del data
	#print train_mean
	#print train_data.shape, len(train_labels)

	params = ['age', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'respire_rate', 'oxy_satur', 'temperature']	#The 7 parameters to be entered by user
	test_data, c_test, inputs, error = np.array([-1 for i in range(len(params))]), 0, {}, ''
	for i in range(len(params)):
		if params[i] in request.GET and request.GET[params[i]]:
			try:
				test_data[i] = float(request.GET[params[i]])
				inputs[params[i]] = test_data[i]
			except ValueError:
				error = 'ValueError. Please enter only integer or floating point values'
				inputs[params[i]] = request.GET[params[i]]
			c_test += 1
		else:
			test_data[i] = train_mean[i]
			inputs[params[i]] = '-'

	if c_test<4:
		error = 'Not enough parameters (at least 4 required)'	#Requires at least 4 out of 8 parameters

	if error:
		return JsonResponse({'Inputs given':inputs, 'Error':error})

	risk = ml_classifiers.solve(train_data, train_labels, test_data)	#Takes the average prediction of 3 state of the art classifiers
	if risk > 0.5:
		pred = 1
	else:
		pred = 0
	f1 = open('static/data2/mortality.data', 'a')		#Writes the test data to the file as a new training sample
	f1.write(','.join([str(a) for a in test_data])+','+str(pred)+'\n')
	f1.close()
	json_data = {'Inputs given':inputs, 'Risk factor':risk}
	return JsonResponse(json_data)
	
	
	#Code to extract features and write them to mortality.data

	"""f1 = open('static/data2/id_age_train.csv', 'r')
	reader = csv.reader(f1)
	data = [row for row in reader]
	data.pop(0)
	del reader
	f1.close()
	train_data = np.array([list(map(int, a)) for a in data])
	#print train_data[0], train_data[1]
	del data

	f1 = open('static/data2/id_label_train.csv', 'r')
	reader = csv.reader(f1)
	data = [row for row in reader]
	data.pop(0)
	del reader
	f1.close()
	train_labels = [list(map(int, a)) for a in data]
	train_labels = np.delete(train_labels, 0, 1)
	#print train_labels[0], train_labels[1]
	del data

	f1 = open('static/data2/id_time_vitals_train.csv', 'r')
	reader = csv.reader(f1)
	data = [row for row in reader]
	data.pop(0)
	del reader
	f1.close()
	for j in range(len(data)):
	    for i in range(len(data[j])):
	    	if data[j][i] == 'NA':
	    		data[j][i] = np.nan
	    	else:
	    		data[j][i] = float(data[j][i])
	vitals = np.array(data)
	del data

	features = []
	for i in range(train_data.shape[0]):
	    a = vitals[vitals[:, 0] == train_data[i][0]]
	    #print len(a)
	    p1 = np.nanmean(a[:, 2:8], axis=0)
	    p2 = np.nanmax(a[:, 2:8], axis=0)
	    p3 = np.nanmin(a[:, 2:8], axis=0)
	    p4 = np.nanstd(a[:, 2:8], axis=0)
	    p5 = p2 - p3
	    features.append(np.concatenate((p1, p2, p3, p4, p5), axis=1))

	train_data = np.concatenate((train_data, np.array(features)), axis=1)
	train_data = np.delete(train_data, 0, 1)
	del features

	imp = Imputer(copy=False)
	imp.fit_transform(train_data)
	print train_data.shape

	f1 = open('static/data2/mortality.data', 'w')
	for i in range(train_data.shape[0]):
		f1.write(','.join([str(a) for a in train_data[i]])+','+str(train_labels[i][0])+'\n')
	f1.close()"""





