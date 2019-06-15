"""
Polynomial linear regression is of the form y = b0 + b1x1 + b2x1^2 + b3x1^3 + ... + bnx1^n
Here we have coefficients for various polynomial degrees and is not a 1 Degree polynomial.
1 Degree polynomial Regression is the same as simple linear regression itself.
Polynimial Regresison can tackle the problems where the predictor variable varies
almost exponentially with respect to the independ variable instead of having a somewhat
linear dependency nature.

In our dataset, we have the salaries for varies Levels in an organization. For different
levels, we try to predict the approximate salary value for the employee.

"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures 

df = pd.read_csv('Position_Salaries.csv')
# print(df.head())
#In our dataset, we need only level an salary as the name 
#of the level is described in the numerical level itself

#Be aware that the independent variables need to be an array
#Only predictor variables need to be a vector
#Hence, use df.iloc[:,1:2].values to get an array of shape (n, 1)
#Array of shape (n,) is not an array but a vector
X = df.iloc[:,1:2].values 
y = df.iloc[:,-1].values

#There is no need to divide our data into train and test split because the data is first of all small


#We create the lienar regression model and fit it to our training data
#The fitted model can be used to predict dependent variable from test data.
print("Fitting the line using simple LinearRegression, we get a bad fit")
linear_regression = LinearRegression()
linear_regression.fit(X, y)

#We first plot the scatter plot for X_train and y_train
print("Plotting the regression line for the training data")
plt.scatter(X, y, color = "red")
#We now plot the regressor line using the model which has the line fitted for the training data
plt.plot(X, linear_regression.predict(X), color = "blue")
plt.title("Salary vs Level of hierarchy")
plt.xlabel("Salary")
plt.ylabel("Level of hierarchy")
plt.show()

#Now let us fit the model using polynomial regression
#We decide a degree for our polynomial fit and add new columns for each degree

poly_regressor = PolynomialFeatures(degree=3)
#We use fit_transform the transform the independent variables to polynomial form
X_poly = poly_regressor.fit_transform(X)

#Now we can see X_poly has a first column filled with 1s (similar to Backward Elimination)
#The next n columns have the value of the independent variable in powers of 1,2,3 ..n

#We now fit our poly_regressor with the newly obtained X_poly
poly_regressor.fit(X_poly)
linear_regression2 = LinearRegression()
linear_regression2.fit(X_poly, y)

print("Plotting the regression curve using Polynomial Regresison for the training data")
plt.scatter(X, y, color = "red")
#We now plot the regressor line using the model which has the line fitted for the training data
#We use poly_regressor.fit_transform(X) as y-axis because we should be able to plot curve for any value of prediction
plt.plot(X, linear_regression2.predict(poly_regressor.fit_transform(X)), color = "blue")
plt.title("Salary vs Level of hierarchy")
plt.xlabel("Salary")
plt.ylabel("Level of hierarchy")
plt.show()

#Will now use our Polynomial regression model to predict the salary of level 6.5
print("Will now use our Polynomial regression model to predict the salary of level 6.5")
print(linear_regression2.predict(poly_regressor.fit_transform(6.5)))