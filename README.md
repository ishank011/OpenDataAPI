##OpenDataAPI
The risk of diabetes in women is significantly increased in case of multiple pregnancies. Early prediction of the same with the help of few measurable parameters can go a long way in taking steps to prevent and counter the disease.

Also, patients admitted to ICUs, in certain cases, have an increased rate of unprecedented deaths.

OpenDataAPI is written in Django to predict the risk factor of diabetes in pregnant women, as well as the chance of mortality in patients admitted to ICU. The API can prove to be of immense use in hospitals as well as private clinics. Pathology centers, which do not offer proper medical advice, can also draw a rough estimate of the conditions of patients using the prediction models.

##Datasets
The dataset for the diabetes prediction model has been taken from Pima Indians Diabetes Database, National Institute of Diabetes and Digestive and Kidney Diseases (1990).
The features used for prediction include:
  1. Number of times pregnant
  2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
  3. Diastolic blood pressure (mm Hg)
  4. Triceps skin fold thickness (mm)
  5. 2-Hour serum insulin (mu U/ml)
  6. Body mass index (weight in kg/(height in m)^2)
  7. Diabetes pedigree function
  8. Age (years)
  9. Class variable (0 or 1)

For the ICU mortality prediction, publicly available data from the Xerox Research Innovation challenge 2015 has been used.
The features are as follows:
  1. Age
  2. Systolic Blood Pressure in mmHg
  3. Diastolic Blood Pressure in mmHg
  4. Heart Rate in bpm
  5. Respiration Rate in bpm
  6. Oxygen Saturation in %
  7. Temperature in Fahrenheit

The features for the same have been extracted using /healthcare/feature_extraction.py

###Usage
The local server can be initialised on a machine running python with the command

    python manage.py runserver

The diabetes API can be accessed using the following url:

    localhost:8000/diabetes_pregnancy/?times_pregnant=a1&glucose_tol=a2&diastolic_bp=a3&triceps=a4&insulin=a5&mass_index=a6&pedigree=a7&agea8

where ai, i=[1,8] are integer or floating point values, and either of which can be left blank.


For accessing the mortality prediction API, the required url is:

    localhost:8000/mortality/?age=a1&systolic_bp=a2&diastolic_bp=a3&heart_rate=a4&respire_rate=a5&oxy_satur=a6&temperature=a7
    
where ai, i=[1,7] are integer or floating point values, and either of which can be left blank.

The data can also be fed to through HTML forms present at the urls diabetes_form and mortality_form respectively.

##Machine Learning techniques
The prediction is made through three state of the art Machine learning classifiers, namely Randomly projected Fischer Linear Discriminant Classifier, Support Vector Machines and Random Forest Classifiers. These are present in /healthcare/ml_classifiers.py

Also, the training set is dynamic in nature, incorporating every valid entry the user enters and using it for future predictions.
