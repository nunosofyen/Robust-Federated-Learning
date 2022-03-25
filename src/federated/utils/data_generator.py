import numpy as np
import tensorflow as tf
import sys
import os
from operator import itemgetter
import random
sys.path.append(os.getcwd() + '/../utils')
from defense import random_defend, random_defend_client


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x, labels, params):
        self.x = x
        self.labels = labels
        self.dim = (params['x_resize_dim'], params['y_resize_dim'])
        self.batch_size = params['batch_size']
        self.n_channels = params['n_channels']
        self.n_classes = params['n_classes']
        self.shuffle = params['shuffle']
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        x_temp = np.asarray([self.x[k] for k in indexes])
        labels_temp = np.asarray([self.labels[k] for k in indexes])

        # Generate data
        # X, y = self.__data_generation(x_temp, labels_temp)

        # return X, y
        return x_temp, labels_temp

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, x_temp, labels_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # y = np.empty((self.batch_size), dtype=int)
        '''
        # Generate data
        for i, _ in enumerate(x_temp):
            # Store sample
            X[i,] = random_defend(x_temp[i])
            # Store class
            y[i] = int(np.where(labels_temp[i]==1)[0][0])
        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
        '''
        # X = random_defend_client(x_temp)
        X = x_temp
        # Should convert the list into numpy array
        return X, labels_temp
