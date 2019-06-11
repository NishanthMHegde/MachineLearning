import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.cross_validation import train_test_split

#import the dataframe
df = pd.read_csv("Data.csv")
print(df.head())
#seperate the independent variables from the dependent variable

#independent variables array
X = df.iloc[:,:-1].values
y = df.iloc[:,3].values 

#We can see that there are a few missing variables.
#Let us use Imputer to assign the mean value of the corresponding column
# to the missing value. axis=0 means columns and axis=1 means rows
# we do this for age and salary which have Nan
imputer = Imputer(missing_values = "NaN", strategy="mean", axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#We can now observe that the Country column is categorical and not numerical
#We need to make our machine learning model understand that it is categorical
#and prevent it from using it for mathematical comparison.
#We first use label encoder to assign integer values to it and then use 
#onehotencoder to split the column into as many columns as the unique values
#and assign a binary value 1 if column value is same as the cateogry, otherwise 0.

label_encoderX = LabelEncoder()
X[:,0] = label_encoderX.fit_transform(X[:,0])
label_encoderY = LabelEncoder()
y = label_encoderX.fit_transform(y)

#onehot encoder is used only for independent categorical features. Here we use first column
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
print(X)
print(y)

#We split our data into 80% training and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#We now need to scale our features (independent variables). This is because
# the difference between two points (x,y) in a graph is not a seen as a simple difference
#in value but is measured in terms of Euclidean Distance between two points. For this purpose
#we need to scale our values so that the Eulidean distances is preserved.
#We scale only our features and not the dependent variables.

scaler = StandardScaler()

#we first fit and transform the training features but not the test features
X_train = scaler.fit_transform(X_train)
#We just transform the test features because we have obtained a fit on the training features
X_test = scaler.transform(X_test)
# We do not scale the test/train dependent variables as they are only used for classification
print(X_train)
print(X_test)
print(y_train)
print(y_test)

