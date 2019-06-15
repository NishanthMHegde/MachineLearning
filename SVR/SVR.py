import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.svm import SVR 
from sklearn.preprocessing import StandardScaler 

df = pd.read_csv('Position_Salaries.csv')
X = df.iloc[:,1:2].values 
y = df.iloc[:,2].values

#For SVR regression, it is compulsory to feature scale the data
std_X = StandardScaler()
std_y = StandardScaler()
X = std_X.fit_transform(X)
y = std_y.fit_transform(y.reshape(-1, 1))

#Create the SVR regression model
#C is a float which tells the threshold amount of error which we can tolerate (default value is 1.0)
#epsilon is max distance of how far away a point can lie outside our SVR 'Street'
#Wthin epsilon range, no penalties are associated. SO better keep it small (default value is 0.1)
svr_model = SVR(C=1.0, kernel='rbf', epsilon=0.1)
svr_model = svr_model.fit(X, y)

print("Plotting the SVR regression line for the training data"
	   "\nPlotted graph is also feature scaled.")
plt.scatter(X, y, color = "red")
#We now plot the regressor line using the model which has the line fitted for the training data
#Plotted graph is also feature scaled.
plt.plot(X, svr_model.predict(X), color = "blue")
plt.title("Salary vs Level of employement")
plt.xlabel("Salary")
plt.ylabel("Level of employement")
plt.show()

#Now let us predict the salary of a level 6.5 employee.
print("Now let us predict the salary of a level 6.5 employee.")

#We need to feature scale 6.5 by converting it to a numpy array
#Then we need to inverse_transform the result to get the actual dollar value.
y_pred = std_y.inverse_transform(svr_model.predict(std_X.transform(np.array([[6.5]]))))
print(y_pred)