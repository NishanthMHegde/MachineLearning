import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
from sklearn.cross_validation import train_test_split
"""
Linear Regression is used for predicting values when one independent variable and
one dependent variable is present. It fits a straight line which is the closest to
each of the points as much as possible. In order to achieve the best-fit, it tries 
to minimize the distance between the points to the regressor line.
For this purpose it makes use of Ordinary Least Square (OLS) method.
OLS method finds the sum of the squares of the difference between the actual value
of the dependent variable for the corresponding independent variable and looks to
minimize this sum of squares between the values. This will give the best fit for the 
training data and hopefully for the test data.

"""
df = pd.read_csv('Salary_Data.csv')
X = df.iloc[:,:-1].values 
y = df.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state=0)

#We create the lienar regression model and fit it to our training data
#The fitted model can be used to predict dependent variable from test data.
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

#We first plot the scatter plot for X_train and y_train
print("Plotting the regression line for the training data")
plt.scatter(X_train, y_train, color = "red")
#We now plot the regressor line using the model which has the line fitted for the training data
plt.plot(X_train, linear_regression.predict(X_train), color = "blue")
plt.title("Salary vs Experience")
plt.xlabel("Salary")
plt.ylabel("Experience")
plt.show()

#We first plot the scatter plot for X_test and y_test
print("Plotting the regression line for the test data")
plt.scatter(X_test, y_test, color = "red")
#We now plot the regressor line using the model which has the line fitted for the training data
#Please note that we should use the line fitted for taining data because this is the line
#which will be used to predict the test data
plt.plot(X_train, linear_regression.predict(X_train), color = "blue")
plt.title("Salary vs Experience")
plt.xlabel("Salary")
plt.ylabel("Experience")
plt.show()