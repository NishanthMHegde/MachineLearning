"""
Logistic Regression is used in cases where we have a Yes/No type of a dependent variable.
It involves problems where we need to predict if a user performed an action like 
whether the user bought a car, passed a test, married, got hired or any situations where
the dependent variable would be a Yes/No or 0/1. 

Logistic Regression works by plotting the same Linear Regression line but the line
is not straight but it looks like the line of Signmoid function (S- shaped).
It predicts the probability of a user having performed a function and it ranges
from 0 to 1 only. 

The equation is also the in the form of y = b0 + b1x1 + ... + bnxn
Here y = ln(1/(p-1)) where p is the probability of having performed an action.
Here,
y = ln(1/(p-1)) 
and p = 1/(1 + e^(-y))

The model works by plotting a line in between (probability of 0.5). All probabilities
below 0.5 are projected to 0 and all probabilities above 0.5 are projected to 1. 

This is a CLASSIFICATION and NOT a REGRESSION
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
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
classifier = LogisticRegression(random_state=0)
classifier = classifier.fit(X_train, y_train)

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
#In this example, out of total 68 positives, 65 samples were correctly predicted as YES and 3 were falsely classified as NO
# Also, out of 32 actual negatives, 8 were falsely classified as positive and 24 were correctly
#classified as negative

#We can summarize it by plotting the classification
print("We can summarize it by plotting the classification")

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()