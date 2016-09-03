# -*- coding: utf-8 -*-

# Code for "TalkingData Mobile User Demographics" competition

import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

events = pd.read_csv("events.csv", dtype={'device_id': np.str})
events["timestamp"]= pd.to_datetime(events["timestamp"], infer_datetime_format=True)
dates = events['timestamp'].dt
events['hour'] = dates.hour
events['day'] = dates.day
data = pd.read_csv("gender_age_train.csv", dtype={'device_id': np.str})
data = pd.merge(events, data, how='left', on='device_id', left_index=True)
del events
data.drop(data[pd.isnull(data['group'])].index, inplace=True)

print data.head()

def create_pivot(attribute1, attribute2, percentage=False):
    group_obj = data.groupby([attribute1, attribute2])['event_id'].count().reset_index()
    pivot_obj = group_obj.pivot(attribute1, attribute2, 'event_id')
    pivot_obj.fillna(0, inplace=True)
    name = "Number of events"
    if percentage:
        name += ' (as percentage of total in period)'
        pivot_obj2 = group_obj.pivot(attribute2, attribute1, 'event_id')
        pivot_obj2 = pivot_obj2/pivot_obj2.sum()
        pivot_obj = pivot_obj2.T
    else:
        max_obj = pivot_obj.values.max()   
        if not max_obj or max_obj == 0:
           return
        pivot_obj = pivot_obj.apply(lambda x: x/max_obj)
        name += ' (absolute normalized values)'
    return pivot_obj, name
    
def plot_heatmap(attribute1, attribute2, percentage=False):
    pivot_obj, name = create_pivot(attribute1, attribute2, percentage)
    plot_core(pivot_obj, attribute1, attribute2, name)
    
def plot_heatmap2(attribute1, attribute2, attribute3='longitude'):
    group_obj = data.groupby([attribute1, attribute2])[attribute3].var().reset_index()
    pivot_obj = group_obj.pivot(attribute1, attribute2, attribute3)
    pivot_obj.fillna(0, inplace=True)
    name = "Mean standard deviation"
    plot_core(pivot_obj, attribute1, attribute2, name)
    
def plot_core(pivot_obj, attribute1, attribute2, name):
    sns.set()
    fig, ax = plt.subplots()
    fig.set_size_inches(1.5 * len(pivot_obj.columns), 0.4 * len(pivot_obj.index))
    ax.set_title(name)
    ax.set_xlabel(attribute1)
    ax.set_ylabel(attribute2)
    sns.heatmap(pivot_obj, linewidths=.5, ax=ax, cmap="PuBu", annot=True, fmt=".2f")
    plt.show()
    
def output_excel(attribute1, attribute2, percentage=False, fillna=True):
    pivot_obj, name = create_pivot(attribute1, attribute2, percentage)
    narep = '0' if fillna else '' 
    if fillna:
        pivot_obj.fillna(0, inplace=True)
    pivot_obj.to_excel(pd.ExcelWriter('device_analysis.xlsx', engine='xlsxwriter'), 
            sheet_name='%s-%s analysis' % (attribute1, attribute2), na_rep=narep)

plot_heatmap('hour', 'gender')
plot_heatmap('hour', 'gender', True)
plot_heatmap('day', 'gender')
plot_heatmap('day', 'gender', True)

bins = [0, 30, 40, 50, 60, 70, 100]
data['age_bins'] = pd.cut(data['age'], bins)
plot_heatmap('hour', 'age_bins')
plot_heatmap('hour', 'age_bins', True)
plot_heatmap('day', 'age_bins')
plot_heatmap('day', 'age_bins', True)

# Longitude and latitude analysis
plot_heatmap2('age_bins', 'gender', 'longitude')
plot_heatmap2('age_bins', 'gender', 'latitude')

# As heatmap has width and height limitations, we output some results to Excel for futher analysis 
output_excel("device_id", "hour", True)
output_excel("device_id", "day", True)
  


