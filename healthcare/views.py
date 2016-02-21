from django.shortcuts import render, render_to_response, get_object_or_404, redirect
from django.http import HttpResponseRedirect, HttpResponse, JsonResponse
import numpy as np
import ml_classifiers


def diabetes_pregnancy(request):
	f1 = open('static/data1/pima-indians-diabetes.data', 'r')		#Data is stored in the given file in static folder
	data = [list(map(float, a.split(','))) for a in f1.readlines()]
	train_labels = [int(a.pop()) for a in data]
	train_data = np.array(data)
	train_mean = np.mean(train_data, axis=0)
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

	if c_test<3:
		error = 'Not enough parameters (at least 3 required)'	#Requires at least 3 out of 8 parameters

	if error:
		return JsonResponse({'Inputs given':inputs, 'Error':error})

	risk = ml_classifiers.solve(train_data, train_labels, test_data)	#Takes the average prediction of 3 state of the art classifiers

	json_data = {'Inputs given':inputs, 'Risk factor':risk}
	return JsonResponse(json_data)




