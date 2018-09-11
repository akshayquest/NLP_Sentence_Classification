"""
@author: AkshayKumar
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Impoering Dataset
dataset = pd.read_csv('DataSet.txt'  , header = None)
z = dataset.iloc[1: , 0].tolist()

for i in range(0, 5485):
    z[i] = z[i].split(' ', 1)

dataset = pd.DataFrame(z)

#Cleanning the text
import re

#To remove unwanted words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, 5485):
    text = re.sub('[^a-zA-Z]', ' ', dataset[1][i]).lower()
    #text = text.split()
    #ps = PorterStemmer()
    #Removing words in for loop, Stemming of words (keeping only roots of words)
    #text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    #text = ' '.join(text)
    corpus.append(text)

#Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 0].values


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

correct = 0

for i in range(0 , 7):
    correct += cm[i][i]

#correct/823
    