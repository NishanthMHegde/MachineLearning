"""
Principal component analysis is used to reduce the number of dimnesions (number of indepenedent varaibles).
If there are N dimensions, then PCA gives K dimensions where K<N but also these K dimensions 
accomodate most of the variance present in the dataset. It is used to remove independent variables
which do not account for most for most of the variance and keeps only the independent variables
which can best explain the variances in the datasets. For example, if there are 3 independent variables,
then PCA reduces the dimensions to 2 so that a 3D graph can be imagined on a 2D plane. 

"""
"""
In our example, we examine variance wines and then decide to which customer segment the wine belongs
to. We also eliminate the less important wines/chemicals. We use Logistic Regression to classify to which
customer segment the wine/alcohols belongs.

PCA needs to be applied right before feature scaling 

"""
import numpy as np 
import pandas as pd 
import math 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import confusion_matrix 
from sklearn.decomposition import PCA 

df = pd.read_csv("Wine.csv")
#Let us use age and salary as independent variables

X = df.iloc[:, 0:13].values
y = df.iloc[:, 13].values 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#Apply PCA here
pca = PCA(n_components=None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
explained_variance = ["%f"%(var) for var in explained_variance]
print(explained_variance)

#From the above results, we can see that ony first 2 independent variables account for most of the variance in the dataset
print("From the above results, we can see that ony first 2 independent variables account for most of the variance in the dataset")

#Let us reduce the number of dimensons to 2
print("Let us reduce the number of dimensons to 2")
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
explained_variance = ["%f"%(var) for var in explained_variance]
print(explained_variance)
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
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()