"""
Natural Language Processing deals with human interaction with the computer and the way
human text or voice is used to collect, analyse, train and predict different things using 
the human text or voice.

NLP can be used for:
a. Predicting the sentiments of a human
b. Predicting the instructions given by humans
c. Creating a chatbot like Siri or Alexa
d. Classifying the reviews based on previous reviews, whether they are positive or negative.

In this example, we will look at implementing a classifier which classifies the reviews
as positive (1) or negative (0) based on previous reviews. The key here is to use 
a training data which consists of words in the previous reviews and how they were classified.

Steps:

1. Clean the input data/text and form a collection of words.
2. Use these collection of words to create a Bag of Words model.
3. Use a machine learning algorithm to train the Bag of Words model.
4. Classify reviews based on the Bag of Words model using the machine learning classification algorithm.


"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix 
from sklearn.cross_validation import train_test_split
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB 
import re

#Let us read our dataset in TSV format because if we use CSV, then pandas will get confused between real commas
#and commas in the reviews.

df = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)

#stopwords contains a list of commonly occuring words like preopositions, nouns, pronouns etc which we dont need
#PorterStemmer is used to remove tenses or verbal form of a word.

corpus = list()
N = df.shape[0]
for i in range(N):
	#Remove non-word characters and replace with a space
	review = re.sub(r'[^a-zA-Z]', ' ', df['Review'][i])
	review = review.lower()
	review = review.split()
	ps = PorterStemmer()
	#stem the word
	review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
	#Join back all the words to form a single string
	review = ' '.join(review)
	corpus.append(review)

#create a count vectorizer whichwill create a one-hot-encoded form of all words present in the list
#Let us use max_features =1500 which will give 1500 independent variables
#We are now creating our bag of words model
print("We are now creating our bag of words model")
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:,1].values 

#Let us choose Naive bayes classification algorithm on our bag of words model.
print("Let us choose Naive bayes classification algorithm on our bag of words model.")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#There is no need to scale our data as we have only 0s, 1s and 2s in our model.
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

print("We can see that we had a decent classification")