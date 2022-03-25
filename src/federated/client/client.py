from os import environ
import flwr as fl
import tensorflow.compat.v1 as tf
import json
import os
from data import Data
import sys
sys.path.append(os.getcwd() + '/../models')
sys.path.append(os.getcwd() + '/../utils')
from adversarial import generate_adversarial_data
from cifar import Cifar
from autoencoder import Autoencoder
from vgg15 import VGG15
from resnet import ResNet
import traceback
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier,TensorFlowClassifier
import numpy as np
import psutil
import gc
from datetime import datetime
import keras
from keras.models import Model
import random

tf.disable_eager_execution()
tf.disable_v2_behavior()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.set_logical_device_configuration(gpu,[tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
    #tf.config.experimental.set_memory_growth(gpu, True)

# Make TensorFlow log less verbose
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
CONFIG_FILE_PATH = "../config.fl.json"
with open(CONFIG_FILE_PATH) as json_data_file:
    configs = json.load(json_data_file)


class Client(fl.client.NumPyClient):
    def __init__(self, model, data):
        self.model = model
        (self.x_train,self.y_train),(self.x_test,self.y_test) = data

    def get_parameters(self):
        # get model weights as numpy array
        return self.model.get_weights()

    def set_parameters(self, parameters):
        # set global model weights to the local model before training for another round
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        # check whether to train from scratch, or resume training with a partially trained model
        with_base_model = configs["fl"]["with_base_model"] == 1
        if (with_base_model == True and config["round"] == 1):
            base_model = keras.models.load_model(configs["fl"]["base_model_path"])
            weights = base_model.get_weights()
        else:  
            weights = parameters
        
        # set the global aggregated weights to the local model before training for another round
        self.model.set_round(config["round"])
        self.model.set_weights(weights)
        history = self.model.fit(
            self.x_train,
            self.y_train,
        )
        # Return updated model parameters and validation results to the FL server
        weights = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0].item(),
            "accuracy": history.history["acc"][0].item(),
            "val_loss": history.history["val_loss"][0].item(),
            "val_accuracy": history.history["val_acc"][0].item(),
        }
        # return the trained weights and the training metrics to the fl server
        return weights, num_examples_train, results

    def evaluate(self, weights, config):
        """Evaluate parameters on the locally held test set."""
        self.model.set_weights(weights)
        # Get config values from the FL server
        steps = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        num_examples_test = len(self.x_test)
        gc.collect()
        return loss, num_examples_test, {"accuracy": accuracy}


def main():
    # get client id from the env variable 
    try:
        client_id = sys.argv[1]
    except:
        try:
            client_id = int(os.environ['CLIENT_ID'])
        except:
            print("NO CLIENT ID GIVEN")
            sys.exit(2)
    
    # load federated data based on the given client id 
    try: 
        data = Data()
        data = data.load_client_data(client_id)
    except :
        traceback.print_exc()
        print("NO DATA FOR SPECIFIED CLIENT ID")
        sys.exit(2)
    
    # compile model
    model_name = configs["client"]["model"]["name"]
    if (model_name == "resnet"):
        model = ResNet()
    elif (model_name == "vgg15"):
        model = VGG15()
    model.compile()
   
   # get server ip and port from config file
    server_ip = configs["server"]["ip"]
    server_port = configs["server"]["port"]

    # initial client and establish connection with the fl server 
    fl.client.start_numpy_client(f"{server_ip}:{server_port}", client=Client(model, data))

if __name__ == "__main__":
    main()
