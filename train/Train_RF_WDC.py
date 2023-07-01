#!/usr/bin/env python
# coding: utf-8

# #IMPORT

# In[30]:


import sys
pythonpath = sys.executable
print(pythonpath)


# In[39]:


import tensorflow as tf
tf.__version__


# In[4]:


import numpy as np
# import pandas as pd
# import datetime
# from datetime import datetime
# import csv
# import os
# from os import listdir
# import json
# import scipy
# import csv
# import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import RNN
# from keras.utils.np_utils import to_categorical
# import keras.backend as K
# from keras import regularizers,optimizers
# from keras.models import load_model
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import RepeatedKFold
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import r2_score
# from sklearn import tree
# from six import StringIO
# from IPython.display import Image
# from sklearn.tree import export_graphviz
# import pydotplus
np.random.seed(2018)


# In[ ]:


# from google.colab import drive
# drive.mount('/content/gdrive')
# drive = '/content/gdrive/My Drive/PaperGiugno/'
# path_db = drive + 'db_blackblaze'


# In[1]:


import sys
pythonpath = sys.executable
print(pythonpath)


# # Functions

# In[6]:


def computeDay(group):
  group = group.sort_values('date')    #ordino in base ai giorni... dal più recente al meno
  group['DayToFailure'] = list(range(group.shape[0]-1, -1,-1 ))
  return group

def divideInLevel(x):
  if x.Label == 0:
    return 'Good' #Good
  elif x.DayToFailure <= 9:
    return 'Alert' # Alert 
  elif x.DayToFailure <= 21:
    return 'Warning ' #Warning 
  else:
    return 'Very Fair'



def tolerance_acc(x):
  if x.pred == 'c_Good':
    return x.vero == 'c_Good' or x.vero == 'c_Very Fair'
  
  if x.pred == 'c_Very Fair':
    return x.vero == 'c_Good' or x.vero == 'c_Very Fair' or x.vero == 'c_Warning'
  
  if x.pred == 'c_Warning':
    return  x.vero == 'c_Very Fair' or x.vero == 'c_Warning' or x.vero == 'c_Alert' 
  
  if x.pred == 'c_Alert':
    return  x.vero == 'c_Warning' or x.vero == 'c_Alert' 


def binary_classification_pred(x):
  if x.pred == 'c_Good'  or x.pred == 'c_Very Fair':
    return 0
  else:
    return 1
    
  
def binary_classification_label(x):
  if x.vero == 'c_Good'  or x.vero == 'c_Very Fair':
    return 0
  else:
    return 1


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
  n_vars = data.shape[1]
  cols, names = list(), list()
  dataclass = data[data.columns[-1:]]
  data = data.drop(columns= ['serial_number', 'Class'], axis = 1)
  columns = data.columns
  # input sequence (t-n, ... t-1)  #non arrivo all'osservazione corrente
  for i in range(n_in-1, 0, -1):
    cols.append(data.shift(i))
    names += [(element + '(t-%d)' % (i)) for element in columns]
    
  for i in range(0, n_out):
    cols.append(data.shift(-i))
    if i == 0:
      names += [(element+'(t)') for element in columns]
    else:
      names += [(element +'(t+%d)' % (i)) for element in columns]
  
  cols.append(dataclass)   #appendo le ultime cinque colonne
  names += ['Class']
    
  agg = pd.concat(cols, axis=1)
  agg.columns = names
  if dropnan:
    agg.dropna(inplace=True)
  
  return agg


def balancing_by_replication(X_train):
  
  alert = X_train[X_train.c_Alert == 1] 
  vfair = X_train[X_train['c_Very Fair'] == 1]
  warn =  X_train[X_train.c_Warning == 1]
  #'c_Alert','c_Good','c_Very Fair','c_Warning'
  good = X_train[X_train.c_Good == 1] # sono i buoni

  size_good = good.shape[0]

  while alert.shape[0] < size_good:
    app = alert.sample(min(alert.shape[0], size_good - alert.shape[0]), replace=False)
    alert = alert.append(app)

  while vfair.shape[0] < size_good:
    app = vfair.sample(min(vfair.shape[0], size_good - vfair.shape[0]), replace=False)
    vfair = vfair.append(app)
  
  while warn.shape[0] < size_good:
    app = warn.sample(min(warn.shape[0], size_good - warn.shape[0]), replace=False)
    warn = warn.append(app)

  
  good = good.append(alert)
  good = good.append(vfair)
  good = good.append(warn)
  return good 


# #Pre- processing

# In[18]:


listLabels = ['c_Alert','c_Good','c_Very Fair','c_Warning']
finestra = 14
# 15model\Train-WDC-WD30EFRX BalckDaUsare

df= pd.read_csv('15model\Train-WDC-WD30EFRX.csv',sep=',')
df = df.dropna()
df.date = pd.to_datetime(df.date, format='%Y-%m-%d').dt.date
df = df.drop(['CurrentPendingSectorCount','ReallocatedSectorsCount','model','capacity_bytes'], axis=1)

scaler = MinMaxScaler(feature_range = (-1,1))
df[['ReportedUncorrectableErrors', 'HighFlyWrites', 'TemperatureCelsius', 
    'RawCurrentPendingSectorCount','RawReadErrorRate', 'SpinUpTime', 
    'RawReallocatedSectorsCount', 'SeekErrorRate', 'PowerOnHours']] = scaler.fit_transform(df[['ReportedUncorrectableErrors', 
                                                                                               'HighFlyWrites', 'TemperatureCelsius', 
                                                                                               'RawCurrentPendingSectorCount',
                                                                                               'RawReadErrorRate', 'SpinUpTime', 
                                                                                               'RawReallocatedSectorsCount', 
                                                                                               'SeekErrorRate', 'PowerOnHours']])

dfHour = df.groupby(['serial_number']).apply(computeDay)
dfHour = dfHour[dfHour.DayToFailure <= 45]
dfHour = dfHour.drop(columns = ['date'])
dfHour['Class'] = dfHour.apply(divideInLevel, axis=1)
dfHour= dfHour.drop(columns= ['Label','DayToFailure', 'serial_number'], axis=1)
dfHour=dfHour.reset_index()
dfHour= dfHour.drop(columns= ['level_1'], axis=1)


# In[19]:


#creo le sequenze
print('Creazione Sequenze')
dfHourSequence =  dfHour.groupby(['serial_number']).apply(series_to_supervised, n_in=finestra, n_out=1, dropnan=True)
dfHourSequence = pd.concat([dfHourSequence, pd.get_dummies(dfHourSequence.Class,prefix='c')], axis=1).drop(['Class'],axis=1)
numberClasses = len(listLabels)

#divisione in train validation e split
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(dfHourSequence[dfHourSequence.columns[:-numberClasses]],
                                                    dfHourSequence[dfHourSequence.columns[-numberClasses:]],
                                                    test_size=0.25,
                                                    random_state=42)

# Split training set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=0.25, # 0.25 x 0.8 = 0.2
                                                  random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(dfHourSequence[dfHourSequence.columns[:-numberClasses]],
#                                                   dfHourSequence[dfHourSequence.columns[-numberClasses:]] ,
#                                                   test_size=0.2, random_state=42)
print(y_train.sum())
print(y_train.columns)
del dfHourSequence
del dfHour


# In[ ]:


# X_val, X_test, y_val, y_test = train_test_split(X_rim, y_rim ,stratify=y_rim, test_size=0.50)
#
#
# del X_rim
# del y_rim
# X_train = pd.concat([X_train, pd.DataFrame(columns = listLabels)], sort = True)
# X_val = pd.concat([X_val,  pd.DataFrame(columns = listLabels)], sort = True)
# X_test = pd.concat([X_test, pd.DataFrame(columns = listLabels)], sort = True)
#
# X_train[listLabels] = y_train.values
# X_val[listLabels] = y_val.values
# X_test[listLabels] = y_test.values


# del y_train
# del y_val
# del y_test
#
#
#
# print('Balancing')
# Complete_train  = balancing_by_replication(X_train)
# print(Complete_train.shape)
# del X_train
#
# print(X_val.groupby(listLabels).count())
# Complete_val = balancing_by_replication(X_val)
# print(Complete_val.shape)
# del X_val

#tolgo le label
# ytrain = Complete_train[listLabels].values
# print(Complete_train[listLabels].sum())
# Xtrain = Complete_train.drop(columns=listLabels, axis=1 )
#
#
# yVal = Complete_val[listLabels].values
# print(Complete_val[listLabels].sum())
# Xval = Complete_val.drop(columns=listLabels, axis=1 )
#
# yTest = X_test[listLabels].values
# Xtest = X_test.drop(columns=listLabels, axis=1 )
#
#
# #reshape come sequenze
# Xtrain = Xtrain.values.reshape(Xtrain.shape[0], finestra, int(Xtrain.shape[1]/finestra))
# Xval = Xval.values.reshape(Xval.shape[0], finestra, int(Xval.shape[1]/finestra))
# Xtest= Xtest.values.reshape(Xtest.shape[0], finestra, int(Xtest.shape[1]/finestra))

# print(Xtrain.shape)
# print(Xval.shape)
# print(Xtest.shape)
#
# print(ytrain.shape)
# print(yVal.shape)
# print(yTest.shape)


# #Modello

# In[21]:


from sklearn.metrics import precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, multilabel_confusion_matrix
from sklearn import metrics
# Create random forest classifier object
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model using the training set
rf.fit(X_train, y_train)

# Predict the response for validation dataset
y_pred_val = rf.predict(X_val)

# Calculate model accuracy on validation set
accuracy_val = accuracy_score(y_val, y_pred_val)

# Print model accuracy on validation set
print("Accuracy on validation set:", accuracy_val)

# Print classification report on validation set
print("Classification Report on validation set:")
print(classification_report(y_val, y_pred_val))

# Plot accuracy curve on validation set
train_scores = []
val_scores = []
for i, estimator in enumerate(rf.estimators_, 1):
    train_scores.append(estimator.score(X_train, y_train))
    X_val_sub = X_val[:min(i*X_val.shape[0]//100+1, 1000)].copy()
    if X_val_sub.shape[0] < 1000:
        X_val_sub = pd.concat([X_val_sub]*(1000//X_val_sub.shape[0]+1), ignore_index=True)[:1000]
    val_scores.append(accuracy_score(y_val[:1000], rf.predict(X_val_sub)))
plt.plot(range(1, 101), train_scores, label='Train')
plt.plot(range(1, 101), val_scores, label='Validation')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Predict the response for test dataset
pred = rf.predict(X_test)

# Calculate model accuracy on test set
accuracy_test = accuracy_score(y_test, pred)
print("Classification Report on test set:")
print(classification_report(y_test, pred))
# Print model accuracy on test set
print("Accuracy on test set:", accuracy_test)
print(pred.shape)
predpd = pd.DataFrame(pred, columns=listLabels)
print(predpd.idxmax(axis=1))
predpd= predpd.idxmax(axis=1)
predpd = predpd.to_frame()

ytestpd = pd.DataFrame(y_test, columns=listLabels)
ytestpd= ytestpd.idxmax(axis=1)

from sklearn import metrics

acc2 = accuracy_score(ytestpd.values, predpd.values)
print('Accuracy sul Test :', acc2)

print('confusion_matrix:')
confm = metrics.confusion_matrix(ytestpd.values, predpd.values)
print(confm)
cm = confm.astype('float') / confm.sum(axis=1)[:, np.newaxis]
print("显示百分比：")
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
print(cm)

print('accuracy_score: ')
print(metrics.accuracy_score(ytestpd.values, predpd.values))
print('precision_score: ')
print(metrics.precision_score(ytestpd.values, predpd.values, average='weighted'))
print('recall_score: ')
print(metrics.recall_score(ytestpd.values, predpd.values, average='weighted'))
print('f1_score: ')
print(metrics.f1_score(ytestpd.values, predpd.values, average='weighted'))
# TN, FP, FN, TP = confm[0,0], confm[0,1], confm[1,0], confm[1,1]
FP = confm.sum(axis=0) - np.diag(confm)
FN = confm.sum(axis=1) - np.diag(confm)
TP = np.diag(confm)
TN = confm.sum() - (FP + FN + TP)
# FP = FP.astype(float)
# FN = FN.astype(float)
# TP = TP.astype(float)
# TN = TN.astype(float)
hh = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)

FPR = FP/(FP+TN)
# print('FPR : ',FPR)
# f1 = lambda FPR:'%.2f%%' % (100*FPR)
# for i in range(len(FPR)):
#     print(FPR[i])
# #     FPR[i] = float(FPR[i]) * 100 + '%'
# #     FPR[i] = str(FPR[i] * 100)[:5].strip()+ '%'
#     FPR[i] = "%.2f%%" % (round((FPR[i]) * 100, 3))
print('FPR : ',FPR)
print('FNR : ',FN/(FN+TP))
MCC = (TP * TN - FP * FN) / np.sqrt(hh)
print('MCC: ', MCC)

c=confusion_matrix(ytestpd.values, predpd.values)
plt.figure(figsize=(12,12))
ax = sns.heatmap(c, yticklabels=1, xticklabels=1, annot=True, fmt="d", cbar=False)
ax.figure.axes[-1].yaxis.label.set_size(20)
ax.set_xlabel("Predicted Label",fontsize=20)
ax.set_ylabel("True Label",fontsize=20)
ax.tick_params(labelsize=13)
ax.set_xticklabels(listLabels)
ax.set_yticklabels(listLabels)
plt.yticks(rotation=0)






# In[9]:





# #Plot

# In[10]:





# #Fine tuning on val

# In[11]:





# #Performance evaluation

# In[12]:





# In[14]:





# In[15]:





# In[16]:





# In[17]:





# In[18]:




