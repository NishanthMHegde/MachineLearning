"""
It is an associative rule learning algorithm. Used for recommendation systems. It is called
Apriori because we try to assume a prior probability.

Terms:

1. Support(M) = (Number of transactions containing M) / (Total number of transactions)

2. Confidence (M1 -> M2) = (Number of transactions containing M1 and M2) / (Total number of transactions containing M1)

2. Lift ((M1 -> M2) = Confidence (M1 -> M2)) / (Support(M2))

Steps:

1. Set a minimum value for support and confidence.

2. Select all subsets in a transation whose support is greater than the minimum support.

3. Select all rules in the subset whose confidence is greater than the minimum confidence.

4. Arrange the rules in descending order of Lift.


Apriori is used to find the most commoly occuring pairs, triplets or a subset of n items in the dataset.
This will give an idea about the retail analysers about how close to keep two sets of items in their
super market or they can even keep the two items far away from each other so that when the buyer travels
across the shopping mall to find the other item, he can also purchase other items on the way.

"""

import numpy as np 
import pandas as pd 
from apyori import apriori 

#Import the dataset and set header=None because in our dataset, there are no headers.

df = pd.read_csv('Market_Basket.csv', header=None)

#We need to convert the dataset into a list of transactions which is a list of list

transactions = list()
for i in range(7501):
	transactions.append([str(df.values[i, j]) for j in range(20)])

#We now need to create the Apriori object and pass in our list of transactions
#We need to also supply value for min_support, min_confidence, min_lift and min_length

#We need an item to appear at least 3 times a week. So we use 3*7=21 as number of appearances
#and 7501 as the total number of transactions. min_support= 0.0028

#We choose a confidence of 20% or 0.2 because we need our pair/triplet or whatever 
#to appear at least 20% of the time in our whole transactions.

#Lift is the probability that the people who liked our first item will also like the second item.
#let us use min_lift of 3

rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)
rules = list(rules)

#We have got a list of rules in descending order of lift. The first rule is the most likeliest and so on.
print("We have got a list of rules in descending order of lift. The first rule is the most likeliest and so on.")
print(rules[0])