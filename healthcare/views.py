from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_protect
from django.http import HttpResponseRedirect, JsonResponse
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

	params = ['times_pregnant', 'glucose_tol', 'diastolic_bp', 'triceps', 'insulin', 'mass_index', 'pedigree', 'age']	#The 8 parameters to be entered by user
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


def diabetes_form(request):
	if request.POST:
		params = ['times_pregnant', 'glucose_tol', 'diastolic_bp', 'triceps', 'insulin', 'mass_index', 'pedigree', 'age']
		l = [a+'='+request.POST[a] for a in params]
		url = '&'.join(l)
		return redirect('/diabetes_pregnancy/?'+url)
	return render(request, 'form1.html')

def mortality_form(request):
	if request.POST:
		params = ['age', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'respire_rate', 'oxy_satur', 'temperature']
		l = [a+'='+request.POST[a] for a in params]
		url = '&'.join(l)
		return redirect('/mortality/?'+url)
	return render(request, 'form2.html')




