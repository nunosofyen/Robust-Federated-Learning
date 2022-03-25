"""Implementation of randomization (resizing and padding) for defending adversarial attck
The following code is revised from https://github.com/anishathalye/obfuscated-gradients/blob/master/randomization/defense.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_soft_device_placement(True)
import json
CONFIG_FILE_PATH = "../config.fl.json"
with open(CONFIG_FILE_PATH) as json_data_file:
    configs = json.load(json_data_file)
modelParams = configs["client"]["model"]["params"]

image_width = modelParams['x_input_dim']
image_height = modelParams['y_input_dim']
image_resize_width = modelParams['x_resize_dim']
image_resize_height = modelParams['y_resize_dim']

PAD_VALUE = 0.5
# inputs should be of shape [batch_size, 373, 64, 1]
def random_defend_client(inputs):
    # At the client level randomlization
    # batch_size = inputs.shape[0]
    batch_size = 16
    rnd_width = tf.random.uniform((), 390, image_resize_width, dtype=tf.int32)
    rnd_height = tf.random.uniform((), 68, image_resize_height, dtype=tf.int32)
    rescaled = tf.image.crop_and_resize(inputs, [[0, 0, 1, 1]] * batch_size, list(range(0, batch_size)), [rnd_width, rnd_height])
    w_rem = tf.subtract(image_resize_width, rnd_width)
    h_rem = tf.subtract(image_resize_height, rnd_height)
    pad_left = tf.random.uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = tf.subtract(w_rem, pad_left)
    pad_top = tf.random.uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = tf.subtract(h_rem, pad_top)
    padded = tf.pad(rescaled, [[0, 0], [pad_left, pad_right], [pad_top, pad_bottom], [0, 0]], constant_values=PAD_VALUE)
    padded.set_shape((batch_size, image_resize_width, image_resize_height, 1))
    return padded



def random_defend_individual(inputs):
    # At the client individual randomlization
    session = tf.compat.v1.Session()
    res = np.zeros((inputs.shape[0], image_resize_width, image_resize_height, 1))
    index = 0
    for x in inputs:
        x = np.expand_dims(x, axis=0)
        rnd_width = tf.random_uniform((), image_width, image_resize_width, dtype=tf.int32)
        rnd_height = tf.random_uniform((), image_height, image_resize_height, dtype=tf.int32)

        rescaled = tf.image.crop_and_resize(x, [[0, 0, 1, 1]], [0], [rnd_width, rnd_height])

        w_rem = image_resize_width - rnd_width
        h_rem = image_resize_height - rnd_height
        pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
        pad_right = w_rem - pad_left
        pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
        pad_bottom = h_rem - pad_top

        padded = tf.pad(rescaled, [[0, 0], [pad_left, pad_right], [pad_top, pad_bottom], [0, 0]], constant_values=PAD_VALUE)
        padded.set_shape((1, image_resize_width, image_resize_height, 1))
        padded  = padded.eval(session=session)
        res[index] = padded
        index += 1
    return res

def random_defend(x, w, h):
    x = np.expand_dims(x, axis=0)
    rnd_width = tf.random.uniform((), image_width-w, image_resize_width, dtype=tf.int32)
    rnd_height = tf.random.uniform((), image_height-h, image_resize_height, dtype=tf.int32)

    rescaled = tf.image.crop_and_resize(x, [[0, 0, 1, 1]], [0], [rnd_width, rnd_height])

    w_rem = image_resize_width - rnd_width
    h_rem = image_resize_height - rnd_height
    pad_left = tf.random.uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    pad_top = tf.random.uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top

    padded = tf.pad(rescaled, [[0, 0], [pad_left, pad_right], [pad_top, pad_bottom], [0, 0]], constant_values=PAD_VALUE)
    padded.set_shape((1, image_resize_width, image_resize_height, 1))
    return padded


def random_layer_batch(inputs):
    # At the client level randomlization
    batch_size = inputs.get_shape().as_list()[0]
    rnd_width = tf.random.uniform((), 373, image_resize_width, dtype=tf.int32)
    rnd_height = tf.random.uniform((), 64, image_resize_height, dtype=tf.int32)
    rescaled = tf.image.crop_and_resize(inputs, [[0, 0, 1, 1]] * batch_size, list(range(0, batch_size)), [rnd_width, rnd_height])
    w_rem = tf.subtract(image_resize_width, rnd_width)
    h_rem = tf.subtract(image_resize_height, rnd_height)
    pad_left = tf.random.uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = tf.subtract(w_rem, pad_left)
    pad_top = tf.random.uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = tf.subtract(h_rem, pad_top)
    padded = tf.pad(rescaled, [[0, 0], [pad_left, pad_right], [pad_top, pad_bottom], [0, 0]], constant_values=PAD_VALUE)
    padded.set_shape((batch_size, image_resize_width, image_resize_height, 1))
    return padded

'''
def random_defend(x, w, h, sess):
    # x = np.expand_dims(x, axis=0)
    batch_size = x.shape[0]
    rnd_width = tf.random.uniform([x.shape[0]], image_width-w, image_resize_width, dtype=tf.int32)
    rnd_height = tf.random.uniform([x.shape[0]], image_height-h, image_resize_height, dtype=tf.int32)

    rescaled = tf.image.crop_and_resize(x, [[0, 0, 1, 1]] * batch_size, list(range(0, batch_size)), [rnd_width, rnd_height])

    w_rem = image_resize_width - rnd_width
    h_rem = image_resize_height - rnd_height
    pad_left = tf.random.uniform([x.shape[0]], 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    pad_top = tf.random.uniform([x.shape[0]], 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top

    padded = tf.pad(rescaled, [[0, 0], [pad_left, pad_right], [pad_top, pad_bottom], [0, 0]], constant_values=PAD_VALUE)
    # padded.set_shape((x.shape[0], image_resize_width, image_resize_height, 1))
    padded  = padded.eval(session=sess)
    return padded
'''
