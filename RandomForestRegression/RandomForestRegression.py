import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor

"""
RandomForest regression makes use of the votes of several decision trees to
calculate the average of a particular interval. It creates a lot more steps
than a single decision tree. This increases the level of accuracy of our 
predictions. 
However, beyond a certain number of estimators, the number of steps in our
graph will not increase by much because of convergence towards a nearly 
optimal prediction. 

Steps: 1. Pick K data points from the training dataset.
	   2. Construct a DecisionTree using the K data points.
	   3. For each data point used for prediction, the accuracy is noted down.
	   4. Repeat steps 1 and 2 till we get the optimal result.

"""
df = pd.read_csv('Position_Salaries.csv')
X = df.iloc[:,1:2].values 
y = df.iloc[:,2].values

#construct the decison tree regressor
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X, y)

#Let us predict the salary of an employee with level of employment of 6.5
print("Let us predict the salary of an employee with level of employment of 6.5")
print(regressor.predict(6.5))

print("Let us now plot a high resolution smoother graph for our non-continous data")
#Let us now plot a high resolution smoother graph
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = "red")
plt.plot(X_grid, regressor.predict(X_grid), color = "blue")
plt.title("Salary vs Level of Employment")
plt.xlabel("Salary")
plt.ylabel("Level of Employment")
plt.show()

#In the above graph, it is a mixture o both horizontal and vertical lines since the data is non-continous.
print("In the above graph, it is a mixture o both horizontal and vertical lines since the data is non-continous.")