import sys
import os
sys.path.append(os.getcwd() + '/../models')
sys.path.append(os.getcwd() + '/../data/demos')
from model import FED_MODEL
import pickle
import json

# read config file
CONFIG_FILE_PATH = "../config.fl.json"
with open(CONFIG_FILE_PATH) as json_data_file:
    configs = json.load(json_data_file)

# get the data for 1 client (centralized learning)
with open('../data/demos/demos_1clients_data.pickle', 'rb') as handle:
    data = pickle.load(handle)
    x_train = data["client_1"]["x_train"]
    print(f" train data shape: ------ {x_train.shape}")
    y_train = data["client_1"]["y_train"]
# build model & fit & save
model = FED_MODEL()
model.compile() 
model.fit(x_train,y_train)
model.save()