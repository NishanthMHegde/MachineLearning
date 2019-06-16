import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeRegressor

"""
Decision Tree creates a split of the dataset based on the independent variables 
based on the entropy and inforomation gain. The spiitting results in the datapoints
being collected over different clusters. The clusters represent some sort of intervals.
The average of each of these intervals is assigned as the value for each interval during
classification/prediction of the dependent variable.

"""
df = pd.read_csv('Position_Salaries.csv')
X = df.iloc[:,1:2].values 
y = df.iloc[:,2].values

#construct the decison tree regressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

#Let us predict the salary of an employee with level of employment of 6.5
print("Let us predict the salary of an employee with level of employment of 6.5")
print(regressor.predict(6.5))

#Let us a plot a graph for DecisionTree using our old methods.
print("Let us a plot a graph for DecisionTree using our old methods.")
plt.scatter(X, y, color = "red")
#We now plot the regressor line using the model which has the line fitted for the training data
plt.plot(X, regressor.predict(X), color = "blue")
plt.title("Salary vs Level of Employment")
plt.xlabel("Salary")
plt.ylabel("Level of Employment")
plt.show()
#We see that the above graph is continous because it was not plotted using intervals.
print("We see that the above graph is continous because it was not plotted using intervals.")

print("Let us now plot a high resolution smoother graph")
#Let us now plot a high resolution smoother graph
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = "red")
plt.plot(X_grid, regressor.predict(X_grid), color = "blue")
plt.title("Salary vs Level of Employment")
plt.xlabel("Salary")
plt.ylabel("Level of Employment")
plt.show()

#In the above graph, it is a mixture of both horizontal and vertical lines since the data is non-continous.
print("In the above graph, it is a mixture of both horizontal and vertical lines since the data is non-continous.")