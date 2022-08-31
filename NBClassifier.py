'''
    EECS 658 - Assignment 01 - NBClassifier
    James Hanselman
    ID: 2976906
'''

# imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#seed
seed = 5

# load iris dataset into numpy dataframe
data = pd.read_csv('iris.csv')

#split data into x & y
y = data.iloc[:, 4] #get all data from fifth column
X = data.iloc[:, :4] #get all data from columns 0-3

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state = seed)

# create model
model = GaussianNB()

# fit data to model (first fold)
model.fit(X_train, y_train)

# create prediction on test data
y_predicted = model.predict(X_test)

# check accuracy, confusion matrix, and other results
accuracy = accuracy_score(y_test, y_predicted)
confusion = confusion_matrix(y_test, y_predicted)
report = classification_report(y_test, y_predicted)

# print results
print(f'Accuracy: {accuracy}\n\nConfusion Matrix:\n{confusion}\n\nP, R, F1 scores:\n{report}')
