"""

KNearestNeighbours is an algoirthm which is used to classify a data point into a class 
given that all other points are classified into classes. The points nearest to the new 
data point determine which class the newly added data point belongs to.

Steps:

1. Select the number of neighbours (K).
2. Pick the K nearest neighbours near the newly added datapoint.
3. Determine the classes to which the K nearest points belong to. Distance is measured
	using Eucledian Distance (Minkowski metric with a power value of 2).
4. Classify the new data point into the class which has the highest 
	occurence among the K nearest neighbours.

"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
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
classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
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
plt.title('KNearestNeighbours (Training set)')
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
plt.title('KNearestNeighbours (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()