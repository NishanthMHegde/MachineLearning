"""
It is an algorithm to cluster the data into groups in which the points belonging to one cluster 
have similar characteristics.

Steps:

1. Decide upon the number of clusters K.
2. Choose K centroids randomly on the graph , one centroid for each cluster which may or may not be part
	of the data points on the scatter plot.
3. The points which are closest to a particular centroid, become a part of that cluster for that centroid. 
	This will give K clusters.
4. Re-compute the centroids by finding the centre of mass for each cluster. If there were re-assignments 
	of points from cluster to another, then repeat step 3 and 4.
5. If no more re-assignments are possible, it means we have arrived at our final model.


Random Initialization trap:

We know that we randomly choose K centroids in the graph. However, we might encounter a situation where
we might choose centroids such that the final model after applying KMeans can give a different clustering
result as opposed to the most optimal cluster. To solve this, we need to use KMeans++ algorithm.

Choosing the right number of clusters:

To arrive at the right number of clusters, we calculate WCSS value for number of clusters starting from 1 
till 10.

WCSS = (sum of distances between the points to centroid in 1st cluster) + (sum of distances between the points to centroid in 2nd cluster) +
(sum of distances between the points to centroid in 3rd cluster) + ... + (sum of distances between the points to centroid in nth cluster) 

inertia is (sum of distances between the points to centroid in (i)th cluster)

The lesser the value of WCSS, the better is our number of clusters. 
If we plot a graph of WCSS and number of clusters, we see that WCSS keeps decreasing. We need to find the ELBOW point
in the graph where the decrease is not substancial. THis will give the correct number of clusters.
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans

df = pd.read_csv('mall.csv')
# print(df.head())

print("Let us choose annual income and spending score as our independent variables")
#Let us choose annual income and spending score as our independent variables

X = df.iloc[:, [3, 4]].values 

wcss = list()

#Let us compute WCSS for cluster number ranging from 1 to 10
#max_iter is the number of maximum re-assignment cycles until model stops.
#n_init is the number of times k-means will be run to select good model.
#inertia is (sum of distances between the points to centroid in (i)th cluster)
for i in range(1,11):
	kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
	kmeans.fit(X)
	wcss.append(kmeans.inertia_)

#Let us plot a curve to calculate the Elbow
print("Let us plot a curve to calculate the Elbow")

plt.plot(range(1,11), wcss)
plt.xlabel("Number of clusters")
plt.ylabel("Intertia")
plt.title("Graph to calculate Elbow")
plt.show()
#Here we see that appropriate number of clusters is 5
print("Here we see that appropriate number of clusters is 5")

#Let us construct a KMeans model with 5 clusters.
print("Let us construct a KMeans model with 5 clusters.")

kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_predict = kmeans.fit_predict(X)
print("The predicted cluster details for our trainind data is")
print(y_predict)

#Let us now plot the scatter plot for the clusterized data using KMeans
print("Let us now plot the scatter plot for the clusterized data using KMeans")

plt.scatter(X[y_predict==0,0], X[y_predict==0,1], s=300, color="red", label="Cluster1")
plt.scatter(X[y_predict==1,0], X[y_predict==1,1], s=300, color="blue", label="Cluster2")
plt.scatter(X[y_predict==2,0], X[y_predict==2,1], s=300, color="green", label="Cluster3")
plt.scatter(X[y_predict==3,0], X[y_predict==3,1], s=300, color="cyan", label="Cluster4")
plt.scatter(X[y_predict==4,0], X[y_predict==4,1], s=300, color="black", label="Cluster5")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, color="yellow", label="centroids")
plt.xlabel("Salary")
plt.ylabel("spending Score")
plt.title("KMeans clustered data")
plt.legend()
plt.show()

print("Now the user has the job to assign some meaningful names to the clusters")
#Now the user has the job to assign some meaningful names to the clusters
plt.scatter(X[y_predict==0,0], X[y_predict==0,1], s=300, color="red", label="Careful customers")
plt.scatter(X[y_predict==1,0], X[y_predict==1,1], s=300, color="blue", label="Standard Customers")
plt.scatter(X[y_predict==2,0], X[y_predict==2,1], s=300, color="green", label="Target Customers")
plt.scatter(X[y_predict==3,0], X[y_predict==3,1], s=300, color="cyan", label="Careless Customers")
plt.scatter(X[y_predict==4,0], X[y_predict==4,1], s=300, color="black", label="Sensible Customers")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, color="yellow", label="centroids")
plt.xlabel("Salary")
plt.ylabel("spending Score")
plt.title("KMeans clustered data")
plt.legend()
plt.show()
