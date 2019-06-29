"""
It is a reinforcement learning algorithm which aims to maximize the reward earned when a particular step is taken.
It is useful in situations where we would like to select a particular item from a set of N items
depending on the actions taken by users/bots. Here, we do not use a training data, but instead carry
out an experiment for a set period of time and then use the results for selecting the best item.
For example, this approach can be used to select the best advertisement logo from a set of similarly
styled experiments.

"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import random 
import math 

#In this example we aim to maximize the reward for advertisements and choose
#the best advertisement
print("In this example we aim to maximize the reward for advertisements and choose"
	  " the best advertisement")
#We are using a CSV dataset just to mimic the user clicking an Advertisement
#It is by no means a training data
df = pd.read_csv("Ads_CTR_Optimisation.csv")

#The dataset has N rounds for each of the D advertisements

#Let us first try the random selection method
print("Let us first try the random selection method")

N = df.shape[0]
d = df.shape[1]
total_reward = 0
ads_selected = []
for n in range(0, N):
	ad = random.randrange(0, d)
	ads_selected.append(ad)
	total_reward = total_reward + df.values[n][ad]

print("Total reward earned by random selection is %s"%(total_reward))
#Let us now visualize the results to see which advertisement performed better
print("Let us now visualize the results to see which advertisement performed better")

plt.hist(ads_selected)
plt.title("Advertisement selection")
plt.xlabel("Advertisement")
plt.ylabel("Frequency of selection")
plt.show()
print("We can see that all the advertisements had similar performance")

#LEt us maximize it by using UCB
print("LEt us maximize it by using UCB")

N = df.shape[0]
d = df.shape[1]
sum_of_rewards = [0]*d
number_of_selections = [0]*d #Number of selections of ads till now 
total_reward = 0
ads_selected = []
for n in range(0, N):
	max_upper_bound = 0
	ad = 0
	for i in range(0, d):
		if number_of_selections[i] > 0:
			average_reward = sum_of_rewards[i] / number_of_selections[i]
			# We use n+1 because index in python start from 0
			delta_i = math.sqrt( 3/2 * math.log(n+1)/number_of_selections[i])
			upper_bound = average_reward + delta_i
		else:
			upper_bound = 1e400 #10 to the power of 44
		if upper_bound > max_upper_bound:
			max_upper_bound = upper_bound
			ad = i

	ads_selected.append(ad)
	number_of_selections[ad] = number_of_selections[ad] + 1
	reward = df.values[n][ad]
	sum_of_rewards[ad] = sum_of_rewards[ad] + reward
	total_reward = total_reward + df.values[n][ad]

#We can see that UCB has better total reward than random selection
print("Total reward earned by UCB is %s"%(total_reward))
print("We can see that UCB has better total reward than random selection")

#Let us now visualize the results to see which advertisement performed better
print("Let us now visualize the results to see which advertisement performed better")

plt.hist(ads_selected)
plt.title("Advertisement selection")
plt.xlabel("Advertisement")
plt.ylabel("Frequency of selection")
plt.show()

#We can now see that 5th advertisement performance better.
#Python indexes start from 0 and hence (4+1)th advertisement can be selected.

print("We can now see that 5th advertisement performance better.")