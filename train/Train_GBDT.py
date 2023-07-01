

import sys
pythonpath = sys.executable
print(pythonpath)


# In[39]:


import tensorflow as tf
tf.__version__


# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
np.random.seed(2018)




import sys
pythonpath = sys.executable
print(pythonpath)


# # Functions

# In[2]:


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

# In[3]:


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


# In[4]:


#creo le sequenze
print('Creazione Sequenze')
dfHourSequence =  dfHour.groupby(['serial_number']).apply(series_to_supervised, n_in=finestra, n_out=1, dropnan=True)
dfHourSequence = pd.concat([dfHourSequence, pd.get_dummies(dfHourSequence.Class,prefix='c')], axis=1).drop(['Class'],axis=1)
numberClasses = len(listLabels)

#divisione in train validation e split
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(dfHourSequence[dfHourSequence.columns[:-numberClasses]],
                                                    dfHourSequence[dfHourSequence.columns[-numberClasses:]],
                                                    test_size=0.2,
                                                    random_state=42)

# Split training set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=0.25, # 0.25 x 0.8 = 0.2
                                                  random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(dfHourSequence[dfHourSequence.columns[:-numberClasses]],
#                                                   dfHourSequence[dfHourSequence.columns[-numberClasses:]] ,
#                                                   test_size=0.2, random_state=42)
print(y_train.sum())
print(X_train.sum())
print(y_train.columns)

print(X_train.columns)
y_train1 = y_train
y_val1 = y_val
y_test1 = y_test

y_train = np.argmax(y_train.values, axis=1)
y_val = np.argmax(y_val.values, axis=1)
y_test = np.argmax(y_test.values, axis=1)
# del dfHourSequence
# del dfHour


# In[ ]:




# #Modello

# In[5]:


from sklearn.metrics import precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, multilabel_confusion_matrix
from sklearn import metrics

# Create GBDT classifier object
gbdt = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Train the model using the training set
gbdt.fit(X_train, y_train)

# Predict the response for validation dataset
y_pred_val = gbdt.predict(X_val)

# Calculate model accuracy on validation set
accuracy_val = accuracy_score(y_val, y_pred_val)

# Print model accuracy on validation set
print("Accuracy on validation set:", accuracy_val)

# Print classification report on validation set
print("Classification Report on validation set:")
print(classification_report(y_val, y_pred_val))

# # Plot accuracy curve on validation set
# train_scores = []
# val_scores = []
# for i, estimator in enumerate(gbdt.estimators_, 1):
#     train_scores.append(estimator.score(X_train, y_train))
#     X_val_sub = X_val[:min(i*X_val.shape[0]//100+1, 1000)].copy()
#     if X_val_sub.shape[0] < 1000:
#         X_val_sub = pd.concat([X_val_sub]*(1000//X_val_sub.shape[0]+1), ignore_index=True)[:1000]
#     val_scores.append(accuracy_score(y_val[:1000], gbdt.predict(X_val_sub)))
# plt.plot(range(1, 101), train_scores, label='Train')
# plt.plot(range(1, 101), val_scores, label='Validation')
# plt.xlabel('Number of Trees')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# Predict the response for test dataset
pred = gbdt.predict(X_test)

# Calculate model accuracy on test set
accuracy_test = accuracy_score(y_test, pred)

# Print model accuracy on test set
print("Accuracy on test set:", accuracy_test)
print("Classification Report on validation set:")
print(classification_report(y_test, pred))
print(pred.shape)

ytestpd, predpd = y_test, pred
confm = metrics.confusion_matrix(ytestpd, predpd)
print(confm)
cm = confm.astype('float') / confm.sum(axis=1)[:, np.newaxis]

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
print(cm)

print('accuracy_score: ')
print(metrics.accuracy_score(ytestpd, predpd))
print('precision_score: ')
print(metrics.precision_score(ytestpd, predpd, average='weighted'))
print('recall_score: ')
print(metrics.recall_score(ytestpd, predpd, average='weighted'))
print('f1_score: ')
print(metrics.f1_score(ytestpd, predpd, average='weighted'))
# TN, FP, FN, TP = confm[0,0], confm[0,1], confm[1,0], confm[1,1]
FP = confm.sum(axis=0) - np.diag(confm)
FN = confm.sum(axis=1) - np.diag(confm)
TP = np.diag(confm)
TN = confm.sum() - (FP + FN + TP)

hh = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)

FPR = FP/(FP+TN)

print('FPR : ',FPR)
print('FNR : ',FN/(FN+TP))
MCC = (TP * TN - FP * FN) / np.sqrt(hh)
print('MCC: ', MCC)

c=confusion_matrix(ytestpd, predpd)
plt.figure(figsize=(12,12))
ax = sns.heatmap(c, yticklabels=1, xticklabels=1, annot=True, fmt="d", cbar=False)
ax.figure.axes[-1].yaxis.label.set_size(20)
ax.set_xlabel("Predicted Label",fontsize=20)
ax.set_ylabel("True Label",fontsize=20)
ax.tick_params(labelsize=13)
ax.set_xticklabels(listLabels)
ax.set_yticklabels(listLabels)
plt.yticks(rotation=0)



