"""

Almost gives same result as KMeans clustering. There are two types of HC, they are Agglomerative HC and Divisive HC.We
shall look at the former.
Steps:
1. Treat each point in the scatter plot as a cluster. SO we have N clusters.
2. Join two of the closest points to make a cluster. We now have N-1 clusters.
3. Join two of the closest clusters. We now have N-2 clusters.
4. Repeat step 3 till we have only one big cluster.
5. Finish!

Distance between clusters can be: Closest points dist, Farthest point dist, Centroid distance, average of all types of distances.
We now have one big cluster. But all the steps we took are recorded in a Dendrogram.

Dendrogram is a graph with X-axis as the points and Y-axis as the Eucledian distance. It has steps for
each step we take. To decide on the optimal number of cluster, we check for the longest vertical line.
FOr this, we need to extend all the horizontal lines of the boxes and then see which line is the longest
vertical line. 
We draw a horizontal cutoff right at the center of that line and see how many vertical line it 
intersects, those many clusters we shall have. The clusters are the result of joining points and clusters
we did in the pevious steps from the cutoff.

"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import scipy.cluster.hierarchy as sch 
from sklearn.cluster import AgglomerativeClustering
df = pd.read_csv('malls.csv')
# print(df.head())

print("Let us choose annual income and spending score as our independent variables")
#Let us choose annual income and spending score as our independent variables

X = df.iloc[:, [3, 4]].values 

#Let us construct a Dendogram first
print("Let us construct a Dendogram first")

#We construct a linkage graph with method=ward which minimizes the variance in the dataset.
dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))
#sch.dendrogram will plot a dendogram
plt.xlabel("Customer points")
plt.ylabel("Eucledian Distance")
plt.title("Dendrogram")
plt.show()
#We can see that the blue line on the right is the longest vertical line even after being cut by horizontal lines in between
print("We can see that the blue line on the right is the longest vertical line even after being cut by horizontal lines in between")

#We can see that when a horizontal line is drawin in between the longest vertical line, it intersects 5 vertical lines
print("We can see that when a horizontal line is drawin in between the longest vertical line, it intersects 5 vertical lines")
#So we need 5 clusters.
print("So we need 5 clusters.")

#Let us construct our Agglomerative Clustering model
print("Let us construct our Agglomerative Clustering model")
hc = AgglomerativeClustering(n_clusters=5, linkage='ward', affinity="euclidean")
y_hc = hc.fit_predict(X)
print(y_hc)

#Let us now plot a scatter plt with the clusters.
plt.scatter(X[y_hc==0,0], X[y_hc==0,1], s=300, color="red", label="Cluster1")
plt.scatter(X[y_hc==1,0], X[y_hc==1,1], s=300, color="blue", label="Cluster2")
plt.scatter(X[y_hc==2,0], X[y_hc==2,1], s=300, color="green", label="Cluster3")
plt.scatter(X[y_hc==3,0], X[y_hc==3,1], s=300, color="cyan", label="Cluster4")
plt.scatter(X[y_hc==4,0], X[y_hc==4,1], s=300, color="black", label="Cluster5")
plt.xlabel("Salary")
plt.ylabel("spending Score")
plt.title("AgglomerativeClustering clustered data")
plt.legend()
plt.show()

print("Now the user has the job to assign some meaningful names to the clusters")
#Now the user has the job to assign some meaningful names to the clusters
plt.scatter(X[y_hc==0,0], X[y_hc==0,1], s=300, color="red", label="Careful customers")
plt.scatter(X[y_hc==1,0], X[y_hc==1,1], s=300, color="blue", label="Standard Customers")
plt.scatter(X[y_hc==2,0], X[y_hc==2,1], s=300, color="green", label="Target Customers")
plt.scatter(X[y_hc==3,0], X[y_hc==3,1], s=300, color="cyan", label="Careless Customers")
plt.scatter(X[y_hc==4,0], X[y_hc==4,1], s=300, color="black", label="Sensible Customers")
plt.xlabel("Salary")
plt.ylabel("spending Score")
plt.title("AgglomerativeClustering clustered data")
plt.legend()
plt.show()