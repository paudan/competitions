# -*- coding: utf-8 -*-

# Code for "TalkingData Mobile User Demographics" competition

import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from zipfile import ZipFile, ZIP_DEFLATED
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.metrics import log_loss
from xgboost.sklearn import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from stacking import Stacking
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.layers.advanced_activations import PReLU

import sys
sys.setrecursionlimit(10000)

datadir = '.'
gatrain = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'),
                      index_col='device_id')
gatest = pd.read_csv(os.path.join(datadir,'gender_age_test.csv'),
                     index_col = 'device_id')
phone = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'))
# Get rid of duplicate device ids in phone
phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')
events = pd.read_csv(os.path.join(datadir,'events.csv'),
                     parse_dates=['timestamp'], index_col='event_id')
appevents = pd.read_csv(os.path.join(datadir,'app_events.csv'), 
                        usecols=['event_id','app_id','is_active'],
                        dtype={'is_active':bool})
applabels = pd.read_csv(os.path.join(datadir,'app_labels.csv'))

gatrain['trainrow'] = np.arange(gatrain.shape[0])
gatest['testrow'] = np.arange(gatest.shape[0])

brandencoder = LabelEncoder().fit(phone.phone_brand)
phone['brand'] = brandencoder.transform(phone['phone_brand'])
gatrain['brand'] = phone['brand']
gatest['brand'] = phone['brand']
Xtr_brand = csr_matrix((np.ones(gatrain.shape[0]), 
                       (gatrain.trainrow, gatrain.brand)))
Xte_brand = csr_matrix((np.ones(gatest.shape[0]), 
                       (gatest.testrow, gatest.brand)))
print('Brand features: train shape {}, test shape {}'.format(Xtr_brand.shape, Xte_brand.shape))

m = phone.phone_brand.str.cat(' '+ phone.device_model)
modelencoder = LabelEncoder().fit(m)
phone['model'] = modelencoder.transform(m)
gatrain['model'] = phone['model']
gatest['model'] = phone['model']
Xtr_model = csr_matrix((np.ones(gatrain.shape[0]), 
                       (gatrain.trainrow, gatrain.model)))
Xte_model = csr_matrix((np.ones(gatest.shape[0]), 
                       (gatest.testrow, gatest.model)))
print('Model features: train shape {}, test shape {}'.format(Xtr_model.shape, Xte_model.shape))

appencoder = LabelEncoder().fit(appevents.app_id)
appevents['app'] = appencoder.transform(appevents.app_id)
napps = len(appencoder.classes_)
deviceapps = (appevents.merge(events[['device_id']], how='left',left_on='event_id',right_index=True)
                       .groupby(['device_id','app'])['app'].agg(['size'])
                       .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                       .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                       .reset_index())
d = deviceapps.dropna(subset=['trainrow'])
Xtr_app = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.app)), 
                      shape=(gatrain.shape[0],napps))
d = deviceapps.dropna(subset=['testrow'])
Xte_app = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.app)), 
                      shape=(gatest.shape[0],napps))
print('Apps data: train shape {}, test shape {}'.format(Xtr_app.shape, Xte_app.shape))

applabels = applabels.loc[applabels.app_id.isin(appevents.app_id.unique())]
applabels['app'] = appencoder.transform(applabels.app_id)
labelencoder = LabelEncoder().fit(applabels.label_id)
applabels['label'] = labelencoder.transform(applabels.label_id)
nlabels = len(labelencoder.classes_)

devicelabels = (deviceapps[['device_id','app']]
                .merge(applabels[['app','label']])
                .groupby(['device_id','label'])['app'].agg(['size'])
                .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                .reset_index())
d = devicelabels.dropna(subset=['trainrow'])
Xtr_label = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.label)), 
                      shape=(gatrain.shape[0],nlabels))
d = devicelabels.dropna(subset=['testrow'])
Xte_label = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.label)), 
                      shape=(gatest.shape[0],nlabels))
print('Labels data: train shape {}, test shape {}'.format(Xtr_label.shape, Xte_label.shape))


Xtrain = hstack((Xtr_brand, Xtr_model, Xtr_app, Xtr_label), format='csr')
Xtest =  hstack((Xte_brand, Xte_model, Xte_app, Xte_label), format='csr')
print('All features: train shape {}, test shape {}'.format(Xtrain.shape, Xtest.shape))

targetencoder = LabelEncoder().fit(gatrain.group)
y = targetencoder.transform(gatrain.group)
nclasses = len(targetencoder.classes_)

gatest_idx = gatest.index
del appevents, applabels, d, deviceapps, devicelabels, events, phone
del gatrain, gatest, m

def score(clf, Xtrain, random_state = 0):
    # Xtrain = Xtrain.toarray()
    X_train, X_valid, y_train, y_valid = train_test_split(Xtrain, y, test_size=0.3)
    print 'Training dataset size: %d, validation dataset size: %d' % \
        (X_train.shape[0], X_valid.shape[0])
    try:
        clf.fit(X_train, y_train)
        return log_loss(y_valid, clf.predict_proba(X_valid)), clf
    except KeyboardInterrupt:
        return None, None
        
    
def create_submission(prediction, score):
    # Make Submission
    now = datetime.datetime.now()
    sub_file = 'submission_' + (str(score) if score else "") + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) 
    print('Writing submission: ', sub_file+ '.csv')
    prediction.to_csv(sub_file + '.csv',index=True)
    zf = ZipFile(sub_file + '.zip', mode='w', compression=ZIP_DEFLATED)
    try:
        zf.write(sub_file + '.csv')
    finally:
        zf.close()
  
#Cs = np.logspace(-6,0,6, base=2)
#res = []
#for C in Cs:
#   res.append(score(LogisticRegression(C = C, multi_class='multinomial',
#                                       solver='newton-cg')))
#plt.semilogx(Cs, res,'-o')
#score(LogisticRegression(C=0.02, multi_class='multinomial',solver='lbfgs'))                        

# Classifier with PCA reduction. PCA did seem not improve results
def classifier_with_pca(Xtrain, y, Xtest, gatest_idx, targetencoder):
    params = {
        "objective": "multi:softprob",    
        "n_estimators": 30,   
        "learning_rate": 0.1,
        "max_depth": 10,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "silent": False,
        # "alpha": 2,
        # "lambda": 5
    }
    clf = XGBClassifier()
    clf.set_params(**params)
    clf.fit(Xtrain, y)
    model = SelectFromModel(clf, prefit=True)
    Xtrain = model.transform(Xtrain)
    Xtest = model.transform(Xtest)
    print 'Dataset size after feature selection: ', Xtrain.shape
    
    pca = PCA(n_components=1000).fit(Xtrain.toarray())
    Xtrain = pca.transform(Xtrain.toarray())
    Xtest = pca.transform(Xtest.toarray())
    print 'Dataset size after principal component extraction: ', Xtrain.shape
    
    # clf = LogisticRegression(C=0.02, multi_class='multinomial',solver='lbfgs')
    clf = XGBClassifier()
    clf.set_params(**params)
    if c_score:
        c_score, clf = score(clf, Xtrain)
        pred = pd.DataFrame(clf.predict_proba(Xtest), 
                            index = gatest_idx, columns=targetencoder.classes_)
        print c_score
        create_submission(pred, c_score)
        del pred

# LogisticRegression classifier, obtaining LS: 2.26508
def logreg_classifier(Xtrain, y, Xtest, gatest_idx, targetencoder):
    clf = LogisticRegression(C=0.02, multi_class='multinomial',solver='lbfgs')
    c_score, clf = score(clf, Xtrain)
    if c_score:
        pred = pd.DataFrame(clf.predict_proba(Xtest), 
                            index = gatest_idx, columns=targetencoder.classes_)
        print c_score
        create_submission(pred, c_score)
        del pred


# define baseline keras model
def baseline_model():    
    model = Sequential()
    model.add(Dense(150, input_dim=Xtrain.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(50, input_dim=Xtrain.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(12, init='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  #logloss
    return model
    

# Keras Deep ANN classifier, obtainining LS: 2.25491 with CV; 2.24565 without CV       
def keras_classifier(Xtrain, y, Xtest, gatest_idx, targetencoder, use_cv=True):
    def batch_generator(X, y, batch_size, shuffle):
        #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
        number_of_batches = np.ceil(X.shape[0]/batch_size)
        counter = 0
        sample_index = np.arange(X.shape[0])
        if shuffle:
            np.random.shuffle(sample_index)
        while True:
            batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
            X_batch = X[batch_index,:].toarray()
            y_batch = y[batch_index]
            counter += 1
            yield X_batch, y_batch
            if (counter == number_of_batches):
                if shuffle:
                    np.random.shuffle(sample_index)
                counter = 0
    
    def batch_generatorp(X, batch_size, shuffle):
        number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
        counter = 0
        sample_index = np.arange(X.shape[0])
        while True:
            batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
            X_batch = X[batch_index, :].toarray()
            counter += 1
            yield X_batch
            if (counter == number_of_batches):
                counter = 0
    
    
    dummy_y = np_utils.to_categorical(y)
    # model=baseline_model()
    # Cross validate NN
    if use_cv:
        kf = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=0)
        pred = np.zeros((y.shape[0],nclasses))
        for itrain, itest in kf:
            model=baseline_model()
            Xtr, Xte = Xtrain[itrain, :], Xtrain[itest, :]
            ytr, yte = dummy_y[itrain], dummy_y[itest]
            fit= model.fit_generator(generator=batch_generator(Xtr, ytr, 400, True),
                             nb_epoch=15,
                             samples_per_epoch=69984,
                             validation_data=(Xte.todense(), yte), verbose=2)
            # evaluate the model
            scores_val = model.predict_generator(generator=batch_generatorp(Xte, 400, False), val_samples=Xte.shape[0])
            pred[itest,:] = scores_val
        c_score = log_loss(y, pred) 
    else:
        model=baseline_model()
        X_train, X_val, y_train, y_val = train_test_split(Xtrain, dummy_y, train_size=0.999, random_state=10)
        fit= model.fit_generator(generator=batch_generator(X_train, y_train, 400, True),
                                 nb_epoch=16,
                                 samples_per_epoch=69984,
                                 validation_data=(X_val.todense(), y_val), verbose=2)
        # evaluate the model
        scores_val = model.predict_generator(generator=batch_generatorp(X_val, 400, False), val_samples=X_val.shape[0])
        c_score = log_loss(y_val, scores_val)   
        del X_train, X_val, y_train, y_val  
    print('logloss val {}'.format(c_score)) 
    scores = model.predict_generator(generator=batch_generatorp(Xtest, 400, False), val_samples=Xtest.shape[0])
    pred = pd.DataFrame(scores, index = gatest_idx, columns=targetencoder.classes_)
    create_submission(pred, c_score)
    del pred
       

# Stacked classifier, resulted in LS: 2.25726
def stacked_classifier(Xtrain, y, Xtest, gatest_idx, targetencoder):
    clfs = [LogisticRegression(C=0.02, multi_class='multinomial',solver='lbfgs'),            
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            LogisticRegression(C=0.5, multi_class='multinomial',solver='lbfgs'),
    ]
    n_folds = 10
    # Generate k stratified folds of the training data.
    skf = list(StratifiedKFold(y, n_folds))
    clf = Stacking(LogisticRegression, clfs, skf, stackingc=False, proba=True) 
    try:
        clf.fit(Xtrain, y)
        # c_score = log_loss(y_valid, clf.predict_proba(X_valid))
        c_score = log_loss(y, clf.predict_proba(Xtrain))
    except KeyboardInterrupt:
        c_score = None
    if c_score:
        pred = pd.DataFrame(clf.predict_proba(Xtest), 
                            index = gatest_idx, columns=targetencoder.classes_)
        print c_score
        create_submission(pred, c_score)
        del pred
    
# stacked_classifier(Xtrain, y, Xtest, gatest_idx, targetencoder)
keras_classifier(Xtrain, y, Xtest, gatest_idx, targetencoder, use_cv=False)
del Xtrain, Xtest