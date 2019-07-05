import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.cross_validation import train_test_split 
import keras
from keras.models import Sequential
from keras.layers import Dense 

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
print("We need to feature scale out training data for Aritifical Neural Networks")

feature_scaler_x = StandardScaler()
X_train = feature_scaler_x.fit_transform(X_train)
X_test = feature_scaler_x.transform(X_test)

#Now let us create our neural network
print("Now let us create our neural network")

#We need to create a Sequential model classifier for our ANN
print("We need to create a Sequential model classifier for our ANN")

classifier = Sequential()

#We now need to add our first hidden layer
print("We now need to add our first hidden/dense layer. Hidden layers will have Relu activation function")
#init=uniform adds uniform weights within 0 to 1
#output_dim is the number of neurons in the hidden layer
#input_dim is the number of input neurons or input observations
#Add Relu activation function on the hidden layers.
classifier.add(Dense(output_dim=6, input_dim=11, init = 'uniform', activation='relu'))

#Let us add one more layer to make it Deep Learning. This time there is no need to add input_dim
print("Let us add one more layer to make it Deep Learning. This time there is no need to add input_dim")
classifier.add(Dense(output_dim=6, init = 'uniform', activation='relu'))

#Now let us add the output layer with sigmoid activation function 
print("Now let us add the output layer with sigmoid activation function")
classifier.add(Dense(output_dim=1, init = 'uniform', activation='sigmoid'))

#Now let us compile the ANN
print("Now let us compile the ANN")
#We use adam optimizer as our stochastic graidet variant, use accuracy metrics and binary_crossentropy
# as our loss calculating function which is similar to the OLS method we used in linear regression.
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])

#We fit our classifier on our training data
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)
y_pred = classifier.predict(y_train)
print(y_pred)
#We got the probability of exiting the bank but we need a classification of yes/no
print("We got the probability of exiting the bank but we need a classification of yes/no")

#We get the classification:
y_pred = (y_pred >0.5)
#We construct a cnfusion metrics
cm = confusion_matrix(y_test, y_pred)
print(cm)
