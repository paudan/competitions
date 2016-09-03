# -*- coding: utf-8 -*-

# Code for "Grupo Bimbo Inventory Demand" competition

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

train_datafile = 'train_data.pkl'

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
      
df_train = load_data()
df_train['log_demand'] = np.log1p(df_train['AdjDemand'])
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
    
df_train['output'] = df_train[['ProductId', 'ClientId']].apply(lambda x: gen_output(x), axis=1)
plt.plot(df_train[['output']])
