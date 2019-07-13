"""
XGBoost is a gradient boosting classification algorithm which uses the concept of DecisionTrees classifier
and the computational power of the CPU to formulate different random samples and pick the 
appropriate classification.
"""

import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier 

df = pd.read_csv("Churn_Modelling.csv")
X = df.iloc[:, 3:13].values 
y = df.iloc[:, 13].values

#We can see that the gender and geography are categorical variables,
#so we label and one_hot encode them
label_encoder_X1 = LabelEncoder()
X[:,1] = label_encoder_X1.fit_transform(X[:,1])
label_encoder_X2 = LabelEncoder()
X[:,2] = label_encoder_X2.fit_transform(X[:,2])

#Let us one hot encode the geography but not the gender because in order to avoid the
#dummy variables trap, we need to remove one column anyways.

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#We need to feature scale out training data for Aritifical Neural Networks
print("We do not need to feature scale out training data for XGBoost")

#Apply XGBoost algorithm here
classifier = XGBClassifier()
classifier = classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

#Let us calculate the accuracy using k-fold cross validation
accuracies = cross_val_score(estimator= classifier, X=X_train, y=y_train, cv=10)
print("The list of accuracies are %s"%(accuracies))
print("The mean of accuracies is %s"%(accuracies.mean()))
print("The standard deviation of accuracies is %s"%(accuracies.std()))

#We had used the same dataset for ANN and it also gave 86% accuracy.
#This is because accuracy also depends on the dataset

print("We had used the same dataset for ANN and it also gave 86% accuracy.")
print("This is because accuracy also depends on the dataset")