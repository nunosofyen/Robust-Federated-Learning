# -*- coding: utf-8 -*-
"""Trains a convolutional neural network on the MNIST dataset, then attacks it with the FGSM attack."""

import keras
from keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
    AveragePooling2D, Reshape, Permute, multiply, ReLU, Dropout
import tensorflow as tf
from keras.utils.layer_utils import get_source_inputs
from keras.utils import layer_utils
from keras import backend as K
from keras.models import Model, Sequential
from keras import layers
from keras import backend as K
import warnings
import json
import pandas as pd
import numpy as np
import random
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier
from art.utils import load_dataset
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
import json
from data_generator import DataGenerator
from art.attacks.evasion import FastGradientMethod
import gc
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import CosineDecay
import math
from math import cos,pi
from optimizers import CustomCosineDecay

class FL_MODEL():
    def __init__(self):
        with open("../config.fl.json") as json_data_file:
            configs = json.load(json_data_file)
        self.configs = configs
        self.batch_size = configs["client"]["model"]["params"]["batch_size"]
        self.epochs = configs["client"]["model"]["params"]["epochs"]
        self.dataset_name = configs["data"]["name"]
        self.n_classes = configs["data"]["classes"]
        self.with_adv_training = configs["fl"]["adv_training"] == 1
        self.attack = configs["client"]["model"]["adversarial"]["attack"]
        self.with_decayed_lr = False
        self.round = 1

    def set_round(self,rnd):
        self.round = rnd

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def evaluate(self, x_test, y_test):
        loss, accuracy = self.model.evaluate(x_test, y_test)
        return loss.item() , accuracy.item()

    def save(self,path=None):
        if path is not None:
            path = path
        else:
            path = "../client/results/"
        n_epochs = self.configs["client"]["model"]["params"]["epochs"]
        model_name = self.configs["client"]["model"]["name"]
        dataset = self.configs["client"]["dataset"]["name"]
        self.model.save(f"{path}/{dataset}-centralized-{model_name}-{n_epochs}epochs")
        
        



