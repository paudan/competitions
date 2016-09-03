# -*- coding: utf-8 -*-

# Code for "TalkingData Mobile User Demographics" competition

import datetime
from timeit import default_timer
import random
from zipfile import ZipFile, ZIP_DEFLATED
import os
import pickle
import time
import shutil
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.sparse import csr_matrix, hstack
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder


def run_xgb(train, test, target):
    params = {
        "objective": "multi:softprob",
        "num_class": 12,
        "booster" : "gbtree",
        "eval_metric": "mlogloss",
        "eta": 0.03,
        "max_depth": 7,
        "subsample": 0.4,
        "colsample_bytree": 0.4,
        "silent": 1,
        # "alpha": 2,
        # "lambda": 5
    }
    X_train, X_valid, y_train, y_valid = train_test_split(train, target, test_size=0.3)
    print 'Training dataset size: ', X_train.shape, \
        'validation dataset size: ', X_valid.shape
    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid = xgb.DMatrix(X_valid, y_valid)

    start_time = default_timer()
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, 10000, evals=watchlist, early_stopping_rounds=20, verbose_eval=True)
    check = gbm.predict(xgb.DMatrix(X_valid))
    score = log_loss(y_valid.tolist(), check)
    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(test))
    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction.tolist(), score

def run_logreg(train, test, target, test_index):
    X_train, X_valid, y_train, y_valid = train_test_split(train, target, test_size=0.3)
    print 'Training dataset size: %d, validation dataset size: %d' % \
        (X_train.shape[0], X_valid.shape[0])
    clf = LogisticRegression(C=10, multi_class='multinomial',solver='lbfgs')
    clf.fit(X_train, y_train)
    pred = pd.DataFrame(clf.predict_proba(test), index = test_index)
    return pred.values.tolist(), log_loss(y_valid, clf.predict_proba(X_valid))

def create_submission(score, test, prediction, test_val):
    # Make Submission
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) 
    print('Writing submission: ', sub_file+ '.csv')
    f = open(sub_file + '.csv', 'w')
    f.write('device_id,F23-,F24-26,F27-28,F29-32,F33-42,F43+,M22-,M23-26,M27-28,M29-31,M32-38,M39+\n')
    total = 0
    for i in range(len(test_val)):
        str1 = str(test_val[i])
        for j in range(12):
            str1 += ',' + str(prediction[i][j])
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()
    zf = ZipFile(sub_file + '.zip', mode='w', compression=ZIP_DEFLATED)
    try:
        zf.write(sub_file + '.csv')
    finally:
        zf.close()


def map_column(table, f):
    labels = sorted(table[f].unique())
    mappings = dict()
    for i in range(len(labels)):
        mappings[labels[i]] = i
    table = table.replace({f: mappings})
    return table
    
def add_count_features(events, output, attribute, byval='event_id', perc_total=True):
    dhour = events.groupby(['device_id', attribute])[byval].count().reset_index()  
    if perc_total:
        dhour = dhour.pivot(attribute, 'device_id', byval)
        dhour.fillna(0, inplace=True)
        dhour = dhour/dhour.sum()
        dhour = dhour.T.reset_index()
    else:
        dhour = dhour.pivot('device_id',attribute,  byval).reset_index()
        dhour.fillna(0, inplace=True)
    dhour.columns = ['%s_%s' % (attribute, x) if x != 'device_id' else 'device_id' for x in dhour.columns.values]
    output = pd.merge(output, dhour, how='left', on='device_id', left_index=True)
    del dhour
    return output

def add_geographical_features(events, output, attribute):
    ev = events[events[attribute] > 0]
    dmean = ev.groupby(['device_id'])[attribute].mean().reset_index() 
    output = pd.merge(output, dmean, how='left', on='device_id', left_index=True)
    dstd = ev.groupby(['device_id'])[attribute].std().reset_index() 
    output = pd.merge(output, dstd, how='left', on='device_id', left_index=True)
    del dmean, dstd, ev
    return output


def get_category(cat):

    def contains_string(strings, source):
        for s in strings:
            if s.lower() in source.lower():
                return True
        return False

    if cat is None or cat.lower() == 'nan':
        return "Other"
    if contains_string(['gambl', 'cards', 'poker'], cat):
        return "Gambling"
    elif contains_string(['game', 'basketball', 'football', 'tennis', 'billards', 'puzzel',
                        'puzzle', 'zombies game', 'warcraft', 'magic', 'dotal-lol',
                        'strategy', 'racing', 'shooting', 'defense'], cat):
        return "Game"
    elif contains_string(['chess'], cat):
        return "Chess"
    elif contains_string(['wig'], cat):
        return "Wig"
    elif contains_string(['comic'], cat):
        return "Comic"
    elif contains_string(['animation'], cat):
        return "Animation"
    elif contains_string(['style', 'education', 'study', 'language', 'exam', 'class'], cat):
        return "Education"
    elif contains_string(['management', 'business', 'account', 'advertis', 'corporate'], cat):
        return "Business"
    elif contains_string(['bank', 'financ', 'insurance', 'securit', 'futures', 
                          'debit', 'credit', 'estate', 'loan', 'card', 'broker',
                          'risk', 'liquidity', 'profitability', 'income', 
                          'payment', 'trust', 'invest', 'shares', 'fund'], cat):
        return "Finance"
    elif contains_string(['mythology', 'painting', 'culture', 'arts', 'museum', 'concert'], cat):
        return "Culture"
    elif contains_string(['book', 'read', 'novel', 'magazine'], cat):
        return "Books"
    elif contains_string(['information', 'news', 'horoscope', 'reviews'], cat):
        return "News"
    elif contains_string(['commun', 'sharing', 'blogs', 'radio', 'music', 
                          'video', 'show', 'im', 'picture'], cat):
        return "Social"
    elif contains_string(['map', 'navigation'], cat):
        return "Maps"
    elif contains_string(['home', 'household',  'decoration', 'appliance', 
                          'furniture', 'clean'], cat):
        return "Home"
    elif contains_string(['health', 'weight', 'medic', 'pharmacy'], cat):
        return "Health"
    elif contains_string(['sports', 'gym', 'fitness', 'swim'], cat):
        return "Sports"
    elif contains_string(['photography'], cat):
        return "Leisure"
    elif contains_string(['job', 'career', 'work'], cat):
        return "Career"
    elif contains_string(['recipes', 'cook', 'food', 'snack', 'food', 'dessert'], cat):
        return "Food & Cooking"
    elif contains_string(['scheduling', 'productivity', 'notes', 'calendar', 'effective'], cat):
        return "Productivity" 
    elif contains_string(['pet', 'cat', 'dog'], cat):
        return "Pets" 
    elif contains_string(['marriage', 'wedding'], cat):
        return "Wedding"  
    elif contains_string(['contact', 'address'], cat):
        return "Contacts"
    elif contains_string(['file', 'disk', 'network', 'engineering', 'system', 
                          'browser', 'desktop', 'utilit', 'editor', 'wifi', ' phone'], cat):
        return "Software"
    elif contains_string(['beauty', 'skin', 'makeup', 'cosmetic', 'make-up',
                          'nail', 'toiletries'], cat):
        return "Beauty"
    elif contains_string(['jewelry', 'fashion', 'women', 'fragrance'], cat):
        return "Fashion"
    elif contains_string(['shoes', 'cloth', 'accessories', 'bag', 'suitcase'], cat):
        return "Clothes"
    elif contains_string(['mother', 'pregnan', 'parent', 'maternal', 'baby',
                          'family', 'families'], cat):
        return "Babies"
    elif contains_string(['camera', 'digital', 'computer', 'phone', 'appliance', 
                          'phones', 'smart '], cat):
        return "Technics"
    elif contains_string(['university', 'college', 'school', 'student'], cat):
        return "Students"
    elif contains_string(['entertainment', 'spa', 'bar', 'club', 'cinema', 'cafe', 
                          'inn', 'dance', 'movie'], cat):
        return "Entertainment"
    elif contains_string(['tobacco'], cat):
        return "Smoking"
    elif contains_string(['service'], cat):
        return "Services"
    elif contains_string(['search'], cat):
        return "Search"
    elif contains_string(['train'], cat):
        return "Training"
    elif contains_string(['shopping', 'services', 'pay', 'buy', 'sell', 'rent', 
                          'retail', 'sale', 'shop', 'product', 'market', 'mall'], cat):
        return "Commerce"
    elif contains_string(['taxi', 'car', 'flight', 'bus', 'train','hotel', 
                          'airport', 'travel', 'tourism', 'reservation', 'tour',
                          'airlines', 'destionation'], cat):
        return "Travel"
    elif contains_string(['trendy'], cat):
        return "Trendy"
    elif contains_string(['simple'], cat):
        return "Trendy"
    elif contains_string(['quality'], cat):
        return "Quality"
    elif contains_string(['passion'], cat):
        return "Passion"
    else:
        return "Other"
        
def create_category_summary():
    labels = pd.read_csv("app_labels.csv", dtype={'app_id': np.str})
    categories = pd.read_csv("label_categories.csv")
    categories['category'] = categories['category'].apply(lambda x: get_category(str(x)))
    categories = pd.merge(categories, labels, how='left', on='label_id', left_index=True)
    del labels
    
    ind = 0
    app_events = pd.DataFrame()
    
    def append_summary_stats(app_events, chunk, categories, ind):
        merged_chunk = pd.merge(chunk, categories, on='app_id', how="inner", left_index=True)
        stats = merged_chunk.groupby(['event_id', 'category'])['label_id'].count()\
                .reset_index().pivot('event_id', 'category', 'label_id')  
        stats.fillna(0, inplace=True)
        app_events = app_events.append(stats)
        ind += len(chunk)
        print 'Processed app_event rows: %d, current length of summary table: %d' % (ind, len(app_events))
        del merged_chunk, stats, chunk
        return app_events, ind
        
    chunksize = 1000000
    remaining = None
    last_id = None
    for chunk in pd.read_csv('app_events.csv', dtype={'app_id': np.str}, chunksize=chunksize):     
        last_id = int(chunk.tail(1)['event_id'])
        remaining_new = chunk[chunk['event_id']==last_id]
        chunk = chunk[chunk['event_id'] != last_id]
        if remaining is not None:
            chunk = remaining.append(chunk)
        remaining = remaining_new  
        app_events, ind = append_summary_stats(app_events, chunk, categories, ind)
        del remaining_new
    # Add last_entry (last_id)
    app_events, ind = append_summary_stats(app_events, remaining, categories, ind)    
    del remaining, categories
    # Remove irrelevant features and rows which contain only zeros
    app_events.drop(['Other'], axis=1, inplace=True)
    app_events = app_events.loc[~(app_events==0).all(axis=1)]
    return app_events.reset_index()


def read_train_test_complex():
    """ Resulted in LS score: 2.22464 with validation data, but very poor on 
    testing, as some devices do not have associated data
    """
    print('Reading category data')
    # Also serialize app_stats data as not to calculate it each time
    pickle_file = 'app_stats.pkl'
    if not os.path.isfile(pickle_file):         
        app_stats = create_category_summary()
        output = open(pickle_file, 'wb')
        pickle.dump(app_stats, output, -1)
    else:
        pfile = open(pickle_file, 'rb')
        app_stats = pickle.load(pfile)
        pfile.close()
        
    print('Reading events data')
    events = pd.read_csv("events.csv", dtype={'device_id': np.str})
    events['counts'] = events.groupby(['device_id'])['event_id'].transform('count')
    events["timestamp"]= pd.to_datetime(events["timestamp"], infer_datetime_format=True)
    dates = events['timestamp'].dt
    events['hour'] = dates.hour
    events['dayofweek'] = dates.dayofweek
    events.drop(['timestamp'], axis=1, inplace=True)
      
    # new_events = pd.DataFrame(events['device_id'].unique(), columns=['device_id'])
    new_events = pd.merge(app_stats, events, how='left', on='event_id', 
                    left_index=True).groupby(['device_id']).sum()
    new_events.drop(['event_id', 'hour', 'dayofweek', 'longitude', 'latitude'], 
                    axis=1, inplace=True)                
    new_events = new_events.reset_index()
    # Add count and geographical features
    new_events = add_count_features(events, new_events, 'hour', perc_total=False)
    new_events = add_count_features(events, new_events, 'dayofweek', perc_total=False)
    new_events = add_geographical_features(events, new_events,'longitude')
    new_events = add_geographical_features(events, new_events,'latitude')
    
    events_total = events[['device_id', 'counts']].drop_duplicates('device_id', keep='first')
    pd.merge(new_events, events_total, how='left', on='device_id', left_index=True)
    del events
    events = new_events
    
    print('Reading device model data')
    pbd = pd.read_csv("phone_brand_device_model.csv", dtype={'device_id': np.str})
    pbd.drop_duplicates('device_id', keep='first', inplace=True)
    pbd['device_model'] = pbd['phone_brand'].str.cat(' '+ pbd['device_model']) 
    pbd = map_column(pbd, 'phone_brand')
    pbd = map_column(pbd, 'device_model')

    print('Reading training data and preparing dataset')
    train = pd.read_csv("gender_age_train.csv", dtype={'device_id': np.str})
    train = map_column(train, 'group')
    train = train.drop(['age', 'gender'], axis=1)
    train = pd.merge(train, pbd, how='left', on='device_id', left_index=True)
    train = pd.merge(train, events, how='inner', on='device_id', left_index=True)
    train.fillna(0, inplace=True)
    # Move group to the end
    labels = train['group']
    Xtr_brand = np.eye(len(pbd.phone_brand.unique())+1)[train['phone_brand']]
    Xtr_model = np.eye(len(pbd.device_model.unique())+1)[train['device_model']]
    train.drop(['device_id', 'group', 'phone_brand', 'device_model'], axis=1, inplace=True)                       
    train = hstack((csr_matrix(train.values.astype(float)), Xtr_brand, 
                    Xtr_model, csr_matrix(np.ones((train.shape[0], 1)))), format='csr')                      
    del Xtr_brand, Xtr_model

    print('Create test dataset...')
    test = pd.read_csv("gender_age_test.csv", dtype={'device_id': np.str})
    test = pd.merge(test, pbd, how='left', on='device_id', left_index=True)
    test = pd.merge(test, events, how='left', on='device_id', left_index=True)
    test.fillna(0, inplace=True)
    test_labels = test['device_id'].values
    test_index = test.index
    Xt_brand = np.eye(len(pbd.phone_brand.unique())+1)[test['phone_brand']]
    Xt_model = np.eye(len(pbd.device_model.unique())+1)[test['device_model']]
    test.drop(['device_id', 'phone_brand', 'device_model'], axis=1, inplace=True)                       
    test = hstack((csr_matrix(test.values.astype(float)), Xt_brand, 
                   Xt_model, csr_matrix(np.ones((test.shape[0], 1)))), format='csr') 
    del Xt_brand, Xt_model    
    return train, test, labels, test_labels, test_index


train, test, labels, test_labels, test_index = read_train_test_complex()
print('Length of train: ', train.shape)
print('Length of test: ', test.shape)
# Logistic regression did not perform successfully,compared to xgboost
#test_prediction, score = run_logreg(train, test, labels, test_index)
test_prediction, score = run_xgb(train, test, labels)
print("LS: {}".format(round(score, 5)))
create_submission(score, test, test_prediction, test_labels)
del test, test_prediction

