"""
Grid search is also a part of model seelction. Grid search is used to find out the optimal
hyperparamters for a particular model that we have selected and have deemed fit to use.
We first select the appropriate model with required accuracy using the Kfold cross-validation
method. Once we have the model nailed down, we can decide the best combination of paramters
that we pass to the model using the GridSearch algorithm. 
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import confusion_matrix 

df = pd.read_csv("Social_Network_Ads.csv")
#Let us use age and salary as independent variables

X = df.iloc[:, [2, 3]].values
y = df.iloc[:, 4].values 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#We need to feature scale age and salary since the model does not automatically take care of it.
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

#Let us now build our model 
#you can play around with the linear, polynomial and rbf Kernels
classifier = SVC(kernel='rbf', random_state=0)
classifier = classifier.fit(X_train, y_train)
#LEt us apply K-fold cross validation to predict how accurate our dataset is
#for all random subsets of our dataset.
print("Let us apply K-fold cross validation to predict how accuracte our dataset is"
	  " for all random subsets of our dataset. ")

#Here estimator is the classifier model and cv is the number of cross fold validations
accuracies = cross_val_score(estimator= classifier, X=X_train, y=y_train, cv=10)
print("The list of accuracies are %s"%(accuracies))
print("The mean of accuracies is %s"%(accuracies.mean()))
print("The standard deviation of accuracies is %s"%(accuracies.std()))

#Apply GridSearch here
print("We apply GridSearch here")

#We need to first construct a list of parameters which we want to test
#Each list item is a dictionary with mapping between parameter name and possible values
#Let us test out the Linear kernel and rbf kernel and see what is best.
#If rbf kernel is used, then we can also test out the appropriate value for Gamma.
#gamma is the coefficient of the values in kernel function 
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
				{'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.5, 0.1, 0.01, 0.001,0.0001]}]
#We construct the GridSearch model and use n_jobs=-1 to use all cores CPU as it is intensive
grid_search = GridSearchCV(estimator = classifier,
							param_grid = parameters,
							cv=2,
							n_jobs =1,
							scoring='accuracy')
grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("We got a best accuracy of %s"%(best_accuracy))
print("We got parameters of %s"%(best_parameters))

#We got rbf kernel, gamma value of 0.5 and a C value of 1. Let us further check the optimal parameters
print("We got rbf kernel, gamma value of 0.5 and a C value of 1. Let us further check the optimal parameters")
parameters = [{'C': [1, 1.5, 2], 'kernel': ['rbf'], 'gamma': [0.5, 0.4, 0.3, 0.6, 0.7, 0.8, 0.9, 1]}]
grid_search = GridSearchCV(estimator = classifier,
							param_grid = parameters,
							cv=2,
							n_jobs =1,
							scoring='accuracy')
grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("We got a best accuracy of %s in second run"%(best_accuracy))
print("We got parameters of %s in second run"%(best_parameters))

#We got a gamma value of 0.6. Let us further try to increase the accuracy by selection a better gamma value.
print("We got a gamma value of 0.6. Let us further try to increase the accuracy by selection a better gamma value.")
parameters = [{'C': [1, 1.5, 2], 'kernel': ['rbf'], 'gamma': [0.55, 0.56, 0.57, 0.58, 0.59, 0.6,0.61, 0.62]}]
grid_search = GridSearchCV(estimator = classifier,
							param_grid = parameters,
							cv=2,
							n_jobs =1,
							scoring='accuracy')
grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("We got a best accuracy of %s in second run"%(best_accuracy))
print("We got parameters of %s in second run"%(best_parameters))
#We can use the best params and further select the best params by narrowing ndown the values to increase accuracy
print("We can use the best params and further select the best params by narrowing ndown the values to increase accuracy")
#Now let us predict the predictor variable values present in the test set
print("Now let us predict the predictor variable values present in the test set")
y_pred = classifier.predict(X_test)
print(y_pred)
#Let us now compare with the actual training data results
print("Let us now compare with the actual training data results")
print(y_test)

#We can see that it is difficult to compare predicted data with real test data
#Let us use confusion matrix to make the comparison simpler 
print("We can see that it is difficult to compare predicted data with real test data")
print("Let us use confusion matrix to make the comparison simpler")
cm = confusion_matrix(y_test, y_pred)
print("The confusion_matrix results are:")
print(cm)
