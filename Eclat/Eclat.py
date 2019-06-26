"""
Eclat is an association rule learning algorithm where we only rely on the 
support of a single/pair/triplet or a set of items in a transactions.
There is no concept of confidence or lift.

Steps:

1. Set a minimum support.
2. Select all sets who have a support greater than the minimum support.
3. Sort the rules in descending order of support.

"""
import numpy as np 
import pandas as pd 
from pyfim import eclat 

#Import the dataset and set header=None because in our dataset, there are no headers.

df = pd.read_csv('Market_Basket.csv', header=None)

#We need to convert the dataset into a list of transactions which is a list of list

transactions = list()
for i in range(7501):
	transactions.append([str(df.values[i, j]) for j in range(20)])