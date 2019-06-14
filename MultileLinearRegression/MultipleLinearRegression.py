"""
Multiple Linear Regression has many independent variables from which we can
predict our predictor/dependent variable.
The equation looks like y = b0 + b1x1 + b2x2 + ... + bnxn
There are different Multiple Linear Regressions:
1. All-in Regression: We consider all the independent variables to fit our model. It is not a 
very good model.
2. Backward Elimination (Fastest and optimal method): a. We first construct a model using all the variables.
						 b. We choose a significance level like 0.05 for exit. 
						 c. We choose a variable whose p-value is greater than our SL. If there
						 is no such variable, then we finalize our model.
						 d. If the variable's p-value is greater than SL, we remote it and
						 fit our model.
						 d. We once again perform step 3 until we have no more variables that can be removed.

3. Forward Selection: a. We construct a model using one variable (Simple linear regression).
					  b. We check if its p-value is less than our SL (entry). If not, we choose another
					  variable.
					  c. We then add another variable to it and construct a model and check its p-value.
					  If it is less than our SL for entry, then we keep it, otherwis we move on to check 
					  with other variables.
					  d. Repear the process till we have no more variables to add or remove.

4. Bi-directional elimination: It is a slower method and takes time.
						a. We start with the first two steps of Forward Selection.
						b. Once we have two variables, we perform Backward Elimination.
						If no variable can be removed, we perform the third step of 
						Forward Selection. 
						c. Once we have moe variables, we continue to perform backward elimination.

5. Score comparison: (Extremely time consuming and not efficient at all)
						a. We construct a model for all possible combination of variables.
						b. We in turn end up getting 2^(n-1) models. For example, if we have
						10 variables, we get 2^(9), i.e, 1023 models.
						c. We choose the model with highest score.
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split 
import statsmodels.api as sm

df = pd.read_csv('50_Startups.csv')
X = df.iloc[:,:-1].values
y = df.iloc[:, 4].values

#The State variable contains info on countries which is a cateogrical variable
#We need to Label Encoe it and onehot encode it.

# One hot encoder is used to create dummy variables for the categorical
#variable. These dummy variable are attached to the beginning of the 
#array.
label_encoder = LabelEncoder()
X[:,3] = label_encoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#We do not need to make use of all dummy variables while fitting the model.
#If we make use of all the dummy variables, then we fall in the trap of dummy variables.
# In the equations y = b0D0 + b1D1 + b2D2, D0 can become 0 if the country is not equal to 1
#and the resulting equation will be b1D1 = y - b2D2. This has created a circular dependency.

X = X[:,1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
#result of simple multiple linear regression
y_pred = linear_regressor.predict(X_test)
print("result of simple multiple linear regression")
print(y_pred)
print("Values used for test")
print(y_test)
print("Just compare both")

#Now let us use backward elimination 

#For backward elimination, we need to add an array of 1s for first column 
#to compensate for the dummy variable that we have removed.
#axis=1 means column
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis=1)

#In the first step, we choose all variables
# Check the constant with highest p-value
print("Summary for first round of backward eliminaton")
X_opt = X[:, [0,1,2,3,4,5]]
estimation = sm.OLS(endog=y, exog=X_opt).fit()
print(estimation.summary())

#We find that x2 constant has highest p-value, so we remove the second variable and fit again.
print("Summary for second round of backward eliminaton")
X_opt = X[:, [0,1,3,4,5]]
estimation = sm.OLS(endog=y, exog=X_opt).fit()
print(estimation.summary())

#we now remove x1 and fit
X_opt = X[:, [0,3,4,5]]
estimation = sm.OLS(endog=y, exog=X_opt).fit()
print(estimation.summary())

#we now remove x2 and fit
X_opt = X[:, [0,3,5]]
estimation = sm.OLS(endog=y, exog=X_opt).fit()
print(estimation.summary())
#We keep doing this untill p-values of all variables are less than or equal to our SL threshold of 0.05
print("We keep doing this untill p-values of all variables are less than or equal to our SL threshold of 0.05")