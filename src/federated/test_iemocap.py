import keras
import tensorflow as tf

config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
sess = tf.compat.v1.Session(config=config)
keras.backend.set_session(sess)

import pickle
import csv
import numpy as np
import sys
import os
import statistics
from sklearn.utils import shuffle
import time
import json
from keras.models import Model
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier
from tensorflow.python.framework.ops import disable_eager_execution
from utils.adversarial import generate_adversarial_data
from utils.score import scoring, write_pre
sys.path.append(os.getcwd() + '/utils')
from defense import random_layer_batch, random_defend, random_defend_client
sys.path.append(os.getcwd() + '/models')
from data_generator import DataGenerator
tf.compat.v1.disable_eager_execution()

CONFIG_FILE_PATH = "./config.fl.json"
with open(CONFIG_FILE_PATH) as json_data_file:
    configs = json.load(json_data_file)
modelParams = configs["client"]["model"]["params"]

with open('./data/iemocap/iemocap_10_s.pickle', 'rb') as handle:
    data = pickle.load(handle)

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

CUR_ADV = 'deepfool'

# for the validation
import math
x_eval = np.array([])
y_eval = np.array([])
for cli_id in range(1, 11):
    train_len = math.ceil(data[f'client_{cli_id}']['x_train'].shape[0] * 0.95)
    _x_eval = data[f'client_{cli_id}']['x_train'][train_len+1:]
    _y_eval = data[f'client_{cli_id}']['y_train'][train_len+1:]
    if cli_id == 1:
        x_eval = _x_eval
        y_eval = _y_eval
    else:
        x_eval = np.concatenate((x_eval, _x_eval), axis=0)
        y_eval = np.concatenate((y_eval, _y_eval), axis=0)
print(x_eval.shape, y_eval.shape)


# for the test
# x_eval = data["x_eval"]
# y_eval = data["y_eval"]
# record_path = 'client/results_iemocap/out_csv/{}'.format(CUR_ADV)
record_path = 'client/results_iemocap/out_csv/fl'
create_folder(record_path)
write_pre(y_eval, os.path.join(record_path, 'truth.csv'))
x_shape = x_eval.shape
rounds = range(150, 310, 10)
csv_path = os.path.join("client/results_iemocap/", 'rand_adv_test.csv')
advs = ['deepfool', 'fgsm', 'pgd']
# advs = ['pgd']
model_base = "client/results_iemocap/models/"
outputs = []
header = ['model', 'attack', 'org', 'adv', 'rand org', 'rand adv']
with open(csv_path, 'a', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
start_time = time.time()

for round_number in rounds:
    # current_model = 'iemocap-resnet-cosine_lr-adv-{}'.format(CUR_ADV)+ '-{}rounds-10clients-1epochs'.format(round_number)
    current_model = 'iemocap-resnet-cosine_lr-' + '{}rounds-10clients-1epochs'.format(round_number)
    create_folder(os.path.join(record_path, current_model))
    MODEL_PATH = model_base + current_model + '.h5'
    model = tf.keras.models.load_model(MODEL_PATH)
    # a = {i: v for i, v in enumerate(model.layers)}
    predictions1 = model.predict(x_eval)
    write_pre(predictions1, os.path.join(record_path, current_model, 'org.csv'))
    result1 = [np.argmax(t) == np.argmax(p) for t, p in zip(y_eval, predictions1)]
    acc0 = sum(result1) / len(result1)

    # layer_name = 'fc7'
    model2= Model(inputs=model.input, outputs=model.get_layer(index=29).output)
    classifier = KerasClassifier(model=model2)
    for adv in advs:
        # if adv not in current_model:
        #       continue
        print('--------------------------------------------------------')
        print('Now dealing with the {} and {}  method'.format(current_model, adv))
        output = [current_model, adv]
        output.append(round(acc0, 4))
        if adv == 'deepfool':
            classifier = KerasClassifier(model=model2)
        else:
            classifier = KerasClassifier(model=model)
        # test on the orignial data and adversarial data
        start_time1 = time.time()
        print('----start adv generation---')
        adv_data = generate_adversarial_data(x_eval, y_eval, classifier, adv)
        print('----finish adv generation: {}'.format(time.time()- start_time1))
        predictions2 = model.predict(adv_data)
        write_pre(predictions2, os.path.join(record_path, current_model, '{}.csv'.format(adv)))
        result2 = [np.argmax(t) == np.argmax(p) for t, p in zip(y_eval, predictions2)]
        acc2 = sum(result2) / len(result2)
        output.append(round(acc2, 4))

        acc1 = []
        acc2 = []
        inputs = tf.compat.v1.placeholder(tf.float32, shape = [modelParams['batch_size'], *x_shape[1:]])
        pad_in = random_layer_batch(inputs)
        sess.run(tf.compat.v1.global_variables_initializer())
        for j in range(3):
            res1 = 0
            res2 = 0
            for i, ((images1, l1), (images2, l2)) in enumerate(zip(DataGenerator(x_eval, y_eval, modelParams), DataGenerator(adv_data, y_eval, modelParams))):
                images1 = sess.run(pad_in, feed_dict={inputs: images1})
                images2 = sess.run(pad_in, feed_dict={inputs: images2})

                predictions1 = model.predict(images1)
                predictions2 = model.predict(images2)
                if i == 0:
                    pre1 = predictions1
                    pre2 = predictions2
                else:
                    pre1 = np.vstack((pre1, predictions1))
                    pre2 = np.vstack((pre2, predictions2))
                res1 += sum([np.argmax(t) == np.argmax(p) for t, p in zip(l1, predictions1)])
                res2 += sum([np.argmax(t) == np.argmax(p) for t, p in zip(l2, predictions2)])
            acc1.append(res1/len(x_eval))
            acc2.append(res2/len(x_eval))
            write_pre(pre1, os.path.join(record_path, current_model, '{}_ran_org_{}.csv'.format(adv, str(j))))
            write_pre(pre2, os.path.join(record_path, current_model, '{}_ran_adv_{}.csv'.format(adv, str(j))))
        output.append(round(statistics.mean(acc1), 4))
        output.append(round(statistics.mean(acc2), 4))
        with open(csv_path, 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(output)
print("--- %s seconds ---" % (time.time() - start_time))
