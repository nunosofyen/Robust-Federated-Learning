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

with open('data/demos/demos_centralized_data.pickle', 'rb') as handle:
    data = pickle.load(handle)

x_eval = data["evaluate"]["x_eval"]
y_eval = data["evaluate"]["y_eval"]
record_path = 'client/results/out_csv'
# write_pre(y_eval, os.path.join(record_path, 'truth.csv'))
x_shape = x_eval.shape
csv_path = os.path.join("client/results/", 'iteration_test.csv')
models = ['dynamic-vgg16-180rounds-68clients-1epochs']
advs = ['deepfool', 'pgd']
iterations = [5, 10, 15, 20, 30, 40, 50]
# models = ['dynamic-adversarial-deepfool-vgg16-220rounds-68clients-1epochs','dynamic-adversarial-fgsm-vgg16-220rounds-68clients-1epochs', 'dynamic-adversarial-pgd-vgg16-220rounds-68clients-1epochs']
# advs = ['fgsm']
outputs = []
header = ['model', 'attack', 'iter', 'org', 'adv', 'rand org', 'rand adv', 'all samples, 3 avg']
with open(csv_path, 'a', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
start_time = time.time()

for m in models:
    MODEL_PATH = "client/results/" + m
    model = tf.keras.models.load_model(MODEL_PATH)
    predictions1 = model.predict(x_eval)
    # write_pre(predictions1, os.path.join(record_path, m, 'org.csv'))
    result1 = [np.argmax(t) == np.argmax(p) for t, p in zip(y_eval, predictions1)]
    acc1 = sum(result1) / len(result1)

    layer_name = 'fc7'
    model2= Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    classifier = KerasClassifier(model=model2)
    for adv in advs:
        for iteration in iterations:
            print('--------------------------------------------------------')
            print('Now dealing with the {} and {} with {} iteration adv method'.format(m, adv, iteration))
            output = [m, adv, iteration]
            output.append(round(acc1, 4))
            if adv == 'deepfool':
                classifier = KerasClassifier(model=model2)
            else:
                classifier = KerasClassifier(model=model)
            # test on the orignial data and adversarial data
            start_time1 = time.time()
            print('----start adv generation---')
            adv_data = generate_adversarial_data(x_eval, y_eval, classifier, adv, iteration)
            print('----finish adv generation: {}'.format(time.time()- start_time1))
            predictions2 = model.predict(adv_data)
            # write_pre(predictions2, os.path.join(record_path, m, '{}.csv'.format(adv)))
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
                # write_pre(pre1, os.path.join(record_path, m, '{}_ran_org_{}.csv'.format(adv, str(j))))
                # write_pre(pre2, os.path.join(record_path, m, '{}_ran_adv_{}.csv'.format(adv, str(j))))
            output.append(round(statistics.mean(acc1), 4))
            output.append(round(statistics.mean(acc2), 4))
            with open(csv_path, 'a', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(output)
print("--- %s seconds ---" % (time.time() - start_time))
