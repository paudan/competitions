# -*- coding: utf-8 -*-
"""
Created on Sun Jun 05 15:07:42 2016

@author: Paulius
"""

__author__ = 'Paulius Danenas'

import numpy as np
import pandas as pd
import operator
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import sklearn.metrics as metrics

import warnings
warnings.filterwarnings('ignore')

def process_dataset(train):
    
    weekdays = {'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday': 4, 'Friday':5, 'Saturday':6, 'Sunday':7}
    train["Dates"]= pd.to_datetime(train["Dates"], infer_datetime_format=True)
    dates = train['Dates'].dt
    train['Year'] = dates.year - 2000
    train['Month'] = dates.month
    train['Day'] = dates.day
    train['Hour'] = dates.hour
    train['Week'] = dates.week
    train.replace(to_replace={'DayOfWeek' : weekdays}, inplace=True)
    enc = LabelEncoder()
    train['PdDistrict'] = enc.fit_transform(train.PdDistrict)
    train['address_type'] = train.Address.apply(lambda x: 1 if '/' in x else 0)
    train['X'] = preprocessing.scale(list(map(lambda x: x+122.4194, train.X)))
    train['Y'] = preprocessing.scale(list(map(lambda x: x-37.7749, train.Y)))
    return train      
    
    
train = pd.read_csv('train.csv', header=0, sep=",", quotechar='"')
train = process_dataset(train)
# Drop possible duplicates (rows that have the same attributes, but different labels)
# Note that it may also remove some classes as well
# X and Y koordinates are excluded from the list of comparison
categories = train['Category'].copy()
label_encoder = LabelEncoder()
train['Category'] = label_encoder.fit_transform(train.Category)
train.drop(["Dates", "Descript", "Resolution", "Address"], axis=1, inplace=True)
cols = train.columns[(train.columns != 'X') & (train.columns != 'Y') & (train.columns != 'Category')]
train.drop_duplicates(subset=cols, keep=False, inplace=True)

test = pd.read_csv('test.csv', header=0, sep=",", quotechar='"')
test = process_dataset(test)
test.drop(["Dates", "Address", "Id"], axis=1, inplace=True)

traindata = np.array(train.ix[:, train.columns != 'Category'])
testdata = np.array(test)
labels = train['Category']
# del train
# del test

print('Data loaded, performing classifier training...')

# Select best features by correlation with output
no_feat = 1   # Seems to perform best with 3 variable, although this selection could be automated
corrs = {}
for i in range(0, traindata.shape[1]):
    corrs[i] = abs(np.corrcoef(traindata[:, i], labels)[0, 1])

best_corrs = sorted(corrs.items(), key=operator.itemgetter(1))
# Highest correlations
best_corrs.reverse()
best_vars = [x[0] for x in best_corrs[:no_feat]]
traindata, testdata = traindata[:, best_vars],  testdata[:, best_vars]

# train.isnull().sum() return 0 for all columns, thus no imputation is needed

# Use hold-out sampling for training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(traindata, labels, test_size=0.3)

# Use Random Forest classifier
# clf=LinearSVC(C=100)
clf = RandomForestClassifier(n_estimators=30)
predictions = clf.fit(X_train, y_train).predict(X_test)
print('Accuracy: %f' % metrics.accuracy_score(y_test, predictions))
print('General F1-score: %f' % metrics.f1_score(y_test, predictions))
print('Classification report')
print(metrics.classification_report(y_test, predictions))

print('Creating CSV file for submission...')

# Get probabilities and form submission dataframe
# Perform training on FULL dataset in order not to miss any class instances which is possible with train-test strategy
clf = RandomForestClassifier(n_estimators=30)
predictions = clf.fit(traindata, labels)
probs = clf.predict_proba(testdata)
submission = pd.DataFrame({label_encoder.inverse_transform([p])[0] : [probs[i][p] for i in range(len(probs))] 
                                                    for p in range(len(probs[0]))})
submission['Id'] = [i for i in range(len(submission))]
for cat in categories:
    if not cat in list(submission.columns):
        submission[cat] = 0.0
submission.reindex_axis(sorted(submission.columns), axis=1)
# Move Id column to the front
cols = submission.columns.tolist()
cols.insert(0, cols.pop(cols.index('Id')))
submission = submission.reindex(columns=cols)
print(submission.columns)
submission.to_csv('submission.csv', index=False)