import math
from random import shuffle
import numpy as np
from numpy import genfromtxt
import os
import pandas as pd
pd.set_option('display.max_rows', 500)
import h5py
import pickle
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize

emos = ['col', 'dis', 'gio', 'pau', 'rab', 'sor', 'tri']
PATH_TO_HDF5_FILE = './logmel_demos.hdf5'

def calculate_scalar(x):
    if x.ndim == 2:
        asix = 0
    elif x.ndim ==3:
        axis = (0, 1)
    mean_val = np.mean(x, axis=axis)
    std_val = np.std(x, axis=axis)

    return mean_val, std_val

def scale(x, mean_val, std_val):
    return (x - mean_val) / std_val


with h5py.File(PATH_TO_HDF5_FILE, "r") as f:
    a_group_key = list(f.keys())
    data = pd.DataFrame()
    for key in a_group_key:
        data[key] = np.array(f[key]).tolist()

for column in data.columns:
    if column != 'logmel':
       data[column] = data[column].apply(lambda x: x.decode('UTF-8'))

# Split the test data as 20% data for each client
evaluationData = data.groupby('speaker_id').sample(frac=0.2, random_state=2)
data=data[~data.isin(evaluationData)].dropna(how = 'all')
all_train_logmel = np.array([x for x in data.logmel])
mean_val, std_val = calculate_scalar(all_train_logmel)
'''
# For statistics
print(data.shape)
print(data['emotion'].value_counts())
print(data.speaker_id.value_counts())

print(evaluationData.shape)
print(evaluationData['emotion'].value_counts())
print(evaluationData.speaker_id.value_counts())
'''
dataDict = {}
x_eval = np.array([x for x in evaluationData.logmel])
x_eval = scale(x_eval, mean_val, std_val)
x_eval = x_eval[:,:,:,np.newaxis]
y_eval = label_binarize([x for x in evaluationData.emotion], classes=emos)
dataDict["x_eval"] = x_eval
dataDict["y_eval"] = y_eval

speakerIDs = list(range(1, 70))
speakerIDs.remove(35)
speakerIDs = list(map(str, speakerIDs))
speakerIDs = ['0' + x if len(x)== 1 else x for x in speakerIDs]
n_clients = 68
shuffle(speakerIDs)
groups = np.array_split(speakerIDs,n_clients)
i=1
for group in groups:
    x_train = np.array([])
    y_train = np.array([])
    for speakerID in group:
        train_data = data[data["speaker_id"]==speakerID]
        _x_train = np.array([v for v in train_data.logmel])
        _x_train = scale(_x_train, mean_val, std_val)
        _x_train = _x_train[:,:,:,np.newaxis]
        _y_train = label_binarize([x for x in train_data.emotion], classes=emos)
        if  (x_train.size == 0):
            x_train = _x_train
            y_train = _y_train
        else:
            x_train = np.concatenate((x_train, _x_train), axis=0)
            y_train = np.concatenate((y_train, _y_train), axis=0)
    dataDict[f"client_{i}"] = {
        "x_train": x_train,
        "y_train": y_train,
    }
    i = i+1

with open(f"demos_{n_clients}clients_data.pickle", "wb") as handle:
    pickle.dump(dataDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
