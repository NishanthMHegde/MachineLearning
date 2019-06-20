"""
Bayes Theorem:

P(A|B) = (P(B|A) * P(A))/P(B)

where P(A|B) = Posterior Probability
	  P(A) = Prior probability
	  P(B) = Marginal likelihood Probability
	  P(B|A) = Maximum likelihood Probability

It is called naive because it just assumes the independence between the variables which
we use as independent variables. 

Scenario: Consider that we have plotted a graph of Age vs Salary of 30 people and we have classified
them into two groups. One group  of 10 people (observations) WALKS while the other group  of 20 DRIVES.
We need to predict that given a new observation or data point X on the graph, whether the new data point
walks or drives.

Approach: We calculate both P(WALKS|X) and P(DRIVES|X) and select the one with greater probability.

Step 1: Calculate P(WALKS|X)
Soln: P(WALKS|X) = (P(X|WALKS|) * P(WALKS))/P(X)

	P(WALKS) = 10/30 = 0.1 
	To calculate P(X) we draw a small circle of some radius from the new data point.
	Let us assume our circle encompasses 4 data points.
	P(X) = Number od data points encompassed/ Total number of observations
	P(X) = 4/30 = 0.13
	To calculate P(X|WALKS) we draw a small circle of some radius from the new data point.
	Let us assume our circle encompasses 4 data points out of which 3 walks and 1 drives.
	P(X|WALKS) = (Number of data points in circle that walk)/(total number of walking data points)
	P(X|WALKS) = 3/10 = 0.3
	P(WALKS|X) = (0.3 * 0.1)/0.13
	P(WALKS|X) = 0.23 = 23% probability of walking

Step 2: Calculate P(DRIVES|X)
		P(DRIVES|X) = (P(X|DRIVES) * P(DRIVES))/P(X)
		P(DRIVES|X) = (1/20 * 0.66)/0.13
		P(DRIVES|X) = 0.25 = 25.3% probability of driving

We see that P(DRIVES|X) > P(WALKS|X), hence we can conclude that the probability of the new point
driving is higher than walking
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
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
#GaussianNB takes no paramters
classifier = GaussianNB()
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