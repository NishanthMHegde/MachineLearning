"""
When SVM classifier is used with the kernel being 'linear', then we call it SVM.
However, when we use 'rbf' Kernel, we call it Kernel SVM because we use a higher dimensional
space to draw a hyperplane(boundary) to classify our points.

When should Kernel SVM be used?
We might encounter instances when we would not be able to to libearly seperate
our data points into classes. This can happen in siutations where if we draw a line
to seperate/classify data points, then we might not end up with all points of one type
on one side of the line.For example, consider the situation where all data points of one
class are situated together in the center and all data points of another class surround the
previous class. In this case there is no way we can linearly seperate them.

Let us try to solve them for 1D and 2D.

1D : Assume the points on the X-axis to be linearly inseperable. In such cases, follow the below steps.
		1. Initial equation is y=X
		2. Convert the equation to y=X-5   (move all points to left by 5 units)
		3. Convert the equation to y=(X-5)^2 (Construct a parabola).
	Now we can easily linearly seperate the data points by cutting the parabola with a line/hyperplane.


2D: Assume the data points of one class are situated together in the center and all data points 
of another class surround the previous class in a 2D plane. Follow the below steps:
	1. Initial equation is f = (x, y)
	2. Introduct a new dimension to make it 3D. Do this by making use of a Kernel function like rbf
		which converts the 2D plane to a Cone like structure with the class of points being present 
		in the center gradually rising to form a cone whereas the class of points surrounding it 
		would have a z=0 height and be on the ground or x-y plane itself.
	3. The resulting equation would be f = (x,y,z).
	4. We can now pass a hyperplane or boundary and linearly seperate the class of points.

Using a kernel SVM can be computationally expensive.
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.svm import SVC
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
#you can play around with the linear, polynomial and rbf Kernels
classifier = SVC(kernel='rbf', random_state=0)
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
plt.title('SVM (Training set)')
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
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()