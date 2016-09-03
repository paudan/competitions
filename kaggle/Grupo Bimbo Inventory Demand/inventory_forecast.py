# -*- coding: utf-8 -*-

# Code for "Grupo Bimbo Inventory Demand" competition

from datetime import datetime
from timeit import default_timer
import operator
import pandas as pd
import numpy as np
import pickle
from zipfile import ZipFile, ZIP_DEFLATED
from xgboost.sklearn import XGBRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import cross_val_score

train_datafile = 'train_data.pkl'

def read_data():
    dtypes = {'Semana': 'int8', 'Agencia_ID': 'int32', 'Canal_ID': 'int8', 'Ruta_SAK': 'int32',
              'Cliente-ID': 'int32', 'Producto_ID': 'int32', 'Venta_hoy': 'float32', 'Venta_uni_hoy': 'int32',
              'Dev_uni_proxima': 'int32', 'Dev_proxima': 'float32', 'Demanda_uni_equil': 'int32'}
    train = pd.read_csv('train.csv', header=0, sep=',', dtype=dtypes)
    rsize = train.values.nbytes / 2 ** 20.
    print("Size for regular values in train dataframe: %.1f MB" % rsize)
    # Serialize dataset to file for faster reloading
    output = open(train_datafile, 'wb')
    # Pickle the list using the highest protocol available.
    pickle.dump(train, output, -1)
    output.close()


def load_data():
    # load the dataset
    pfile = open(train_datafile, 'rb')
    df_train = pickle.load(pfile)
    pfile.close()
    df_train.columns = ['WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId', \
                        'ProductId', 'SalesUnitsWeek', 'SalesPesosWeek', \
                        'ReturnsUnitsWeek', 'ReturnsPesosWeek', 'AdjDemand']
    print df_train.head()
    return df_train


def compute_naive_model(df_train, use_reg=False):
    # Computing the medians by grouping the entire data on ProductId and then grouping on both ProductId and the ClientId.
    # Score:0.500086;if linear regression is used to adjust coefficients: 0.51321
    prod_tab = df_train.groupby('ProductId').agg({'log_demand': np.mean})
    prod_tab2 = df_train.groupby(['ProductId', 'ClientId']).agg({'log_demand': np.mean})
    global_val = np.expm1(np.mean(df_train['log_demand']))
    prod_dict2 = prod_tab2.to_dict()
    prod_dict = prod_tab.to_dict()

    def gen_output(key):
        key = tuple(key)
        try:
            try:
                val = np.expm1(np.mean(prod_dict2['log_demand'][key]))
            except:
                val = np.expm1(np.mean(prod_dict['log_demand'][key[0]]))
        except:
            val = global_val
        return val
    
    if use_reg:
        df_train['output'] = df_train[['ProductId', 'ClientId']].apply(lambda x: gen_output(x), axis=1)
        X_train = df_train[['output']]
        y_train = df_train[['AdjDemand']]
        print X_train.shape, y_train.shape
        del df_train        
        reg = LinearRegression()        
        reg.fit(X_train, y_train)
        print reg.intercept_[0], reg.coef_[0][0] 
        del X_train, y_train                 
        df_test = pd.read_csv('test.csv')
        df_test.columns = ['id', 'WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId', 'ProductId']
        # Generating the output
        df_test['Demanda_uni_equil'] = df_test[['ProductId', 'ClientId']]\
            .apply(lambda x: reg.intercept_[0]  + gen_output(x) * reg.coef_[0][0], axis=1)
    else:
        df_test = pd.read_csv('test.csv')
        df_test.columns = ['id', 'WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId', 'ProductId']
        # Generating the output
        df_test['Demanda_uni_equil'] = df_test[['ProductId', 'ClientId']].apply(lambda x: gen_output(x), axis=1)
    return df_test
    
def naive_regression(df_train):
    prod_tab = df_train.groupby('ProductId').agg({'log_demand': np.mean})
    prod_tab2 = df_train.groupby(['ProductId', 'ClientId']).agg({'log_demand': np.mean})
    global_val = np.expm1(np.mean(df_train['log_demand']))
    prod_dict2 = prod_tab2.to_dict()
    prod_dict = prod_tab.to_dict()
    df_train    


def create_submission(df_test):
    df_submit = df_test[['id', 'Demanda_uni_equil']]
    df_submit = df_submit.set_index('id')
    sub_file = 'submission_' + str(datetime.now().strftime("%Y-%m-%d-%H-%M"))
    df_submit.to_csv(sub_file + '.csv')
    zf = ZipFile(sub_file + '.zip', mode='w', compression=ZIP_DEFLATED)
    try:
        zf.write(sub_file + '.csv')
    finally:
        zf.close()


def train_predictor(train_df):
    # Kaggle score: 0.72120
    attributes = ['WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId', 'ProductId']
    labels = np.array(train_df['AdjDemand'])
    temp_df = train_df[attributes]
    del train_df
    traindata = np.array(temp_df)
    del temp_df

    print 'Creating training and validation datasets...'
    # Use hold-out sampling for training and testing datasets
    X_train, X_val, y_train, y_val = train_test_split(traindata, labels, test_size=0.3)
    del traindata
    del labels
    print 'Size of training dataset: %d. validation dataset: %d' % (len(X_train), len(X_val))

    print 'Building regression model...'
    start_time = default_timer()
    # rfreg = RandomForestRegressor(oob_score=True, verbose=1)
    rfreg = XGBRegressor(n_estimators=30, silent=False, objective="reg:linear", max_depth=10)
    selected_indices = None
    if not isinstance(rfreg, XGBRegressor):
        rfreg = rfreg.fit(X_train, y_train, verbose=True, early_stopping_rounds=20)
        model = SelectFromModel(rfreg, prefit=True)
        selected_indices = model.get_support()
        print "Selected attributes:", selected_indices
        # Retrain predictor with selected features
        X_selected = model.transform(X_train)
        X_val_selected = model.transform(X_val)
        rfreg = rfreg.fit(X_selected, y_train)
        y_output = rfreg.predict(X_val_selected)
    else:
        rfreg = rfreg.fit(X_train, y_train)
        y_output = rfreg.predict(X_val)
    mse_rf = mean_squared_error(y_val, y_output)
    print "MSE obtained with the validation dataset:", mse_rf
    print 'Total time for model training:',  default_timer() - start_time

    # Perform prediction
    df_test = pd.read_csv('test.csv')
    testdata = np.array(df_test.ix[:, 1:])
    if selected_indices:
        testdata = testdata.ix[:, selected_indices]
    df_test['Demanda_uni_equil'] = [max(0, pred) for pred in rfreg.predict(testdata)]
    return df_test


df_train = load_data()
df_train['log_demand'] = np.log1p(df_train['AdjDemand'])
#df_test = train_predictor(df_train)
#print df_test.head()
#create_submission(df_test)
results = compute_naive_model(df_train, use_reg=True)
create_submission(results)
