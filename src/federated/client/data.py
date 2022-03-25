import tensorflow as tf 
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os
#from art.utils import load_dataset
import pickle
sys.path.append(os.getcwd() + '/../utils/')
from defense import random_defend_client, random_defend_individual
import numpy as np
sys.path.append(os.getcwd() + '/../data/lenze')
sys.path.append(os.getcwd() + '/../data/bosch')
sys.path.append(os.getcwd() + '/../data/demos')

class Data():
    def __init__(self):
        CONFIG_FILE_PATH = "../config.fl.json"
        with open(CONFIG_FILE_PATH) as json_data_file:
            self.configs = json.load(json_data_file)
            self.dataset = self.configs["data"]["name"]
        
    def make_federated_data(self, n_clients, fl_type = "horizontal"):
        if (fl_type == "horizontal"):
            pass
        elif (fl_type == "vertical"):
            pass
        
    def preprocess(self,data):
        pass
        
    def load_client_data(self,client_id):
        if(self.dataset == "cifar10"):
            return tf.keras.datasets.cifar10.load_data()
        elif (self.dataset == "lenze"):
            ml_data = pd.read_pickle('../data/lenze/ML_Data.pickle')
            signal_data = pd.read_pickle('../data/lenze/Signal_Data.pickle').drop(['RR_scaled_scaled',"RR71P_scaled_scaled"],axis=1)
            meta_data = pd.read_pickle('../data/lenze/Meta_Data.pickle')
            data = pd.concat([signal_data, meta_data,ml_data], axis=1)
            return data
        
        elif(self.dataset == "demos"):
            num_clients = self.configs["fl"]["num_clients"]
            with open(f"../data/demos/demos_16clients_data.pickle", 'rb') as handle:
                demos_data = pickle.load(handle)
                data = demos_data[f"client_{client_id}"]
                handle.close()
            x_train = data['x_train'].astype(np.float32)
            y_train = data['y_train'].astype(np.float32)
            x_test = demos_data['x_eval'].astype(np.float32)
            y_test = demos_data['y_eval'].astype(np.float32)
            
            return (x_train,y_train),(x_test, y_test)
        
        elif(self.dataset == "iemocap"):
            num_clients = self.configs["fl"]["num_clients"]
            with open(f"../data/iemocap/iemocap_{num_clients}clients_data.pickle", 'rb') as handle:
                demos_data = pickle.load(handle)
                data = demos_data[f"client_{client_id}"]
                handle.close()
            x_train = data['x_train'].astype(np.float32)
            y_train = data['y_train'].astype(np.float32)
            x_test = demos_data['x_eval'].astype(np.float32)
            y_test = demos_data['y_eval'].astype(np.float32)
            
            return (x_train,y_train),(x_test, y_test)