import sys
import os
import numpy as np
sys.path.append(os.getcwd() + '/../models')
sys.path.append(os.getcwd() + '/../data/demos')
sys.path.append(os.getcwd() + '/../utils')
from model import FED_MODEL
from models import basic_CNN, VGG16
import pickle
import json

import argparse
from tensorflow import keras
import pickle
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier
from tensorflow.python.framework.ops import disable_eager_execution
from adversarial import generate_adversarial_data
disable_eager_execution()

parser = argparse.ArgumentParser(description='Adversarial Fine-tunning.')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--attack', type=str, choices=['fgsm', 'deepfool', 'pgd'], default='fgsm')
parser.add_argument('--savepath', type=str, default='./adv_results/')
args = parser.parse_args()

lr = args.learning_rate
attack_method = args.attack
save_path = args.savepath

CONFIG_FILE_PATH = "../config.fl.json"
with open(CONFIG_FILE_PATH) as json_data_file:
    configs = json.load(json_data_file)
modelParams = configs["client"]["model"]["params"]
model_name = configs["client"]["model"]["name"]

# get the data
with open('../data/demos/demos_centralized_data.pickle', 'rb') as handle:
    data = pickle.load(handle)
    x_train = data["train"]["x_train"]
    y_train = data["train"]["y_train"]

# model = model = VGG16(input_shape=(373,64,1))
model = basic_CNN()
model.compile(optimizer=keras.optimizers.Adam(lr),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# prepare the adversarilized data to fine-tune the data 
classifier = KerasClassifier(model=model)
x_adv_data = generate_adversarial_data(x_train, classifier, attack_method)

# concatenate the adversarial data and original data
x_train = np.concatenate((x_train, x_adv_data), axis=0)
y_train = np.concatenate((y_train, y_train), axis=0)

model.fit(x_train, y_train, batch_size=modelParams["batch_size"], epochs=modelParams["epochs"], validation_split=0.1, verbose=1)
model.save(save_path + '/' + model_name + '_' + str(lr) + '_' + attack_method)
