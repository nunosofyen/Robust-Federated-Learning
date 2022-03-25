from keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
    AveragePooling2D, Reshape, Permute, multiply, ReLU, Dropout
from keras.utils.layer_utils import get_source_inputs
from keras import backend as K
from keras.models import Model, Sequential
from keras import backend as K
import numpy as np
import random
from art.estimators.classification import KerasClassifier
from tensorflow.python.framework.ops import disable_eager_execution
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from math import cos,pi
from optimizers import CustomCosineDecay
from model import FL_MODEL
disable_eager_execution()

class ResNet(FL_MODEL):
    def compile(self, input_tensor=None, input_tensor=None, input_shape=None, pooling=None):
        img_input = Input(shape=input_shape)
        # Block 1
        x = Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

        # Block 2
        identity = x
        x = Conv2D(128, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        identity = Conv2D(128, kernel_size=1, strides=1, padding='valid', use_bias=False)(identity)
        identity = BatchNormalization()(identity)
        x += identity
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2))(x)

        # Block 3
        identity = x
        x = Conv2D(256, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(256, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        identity = Conv2D(256, kernel_size=1, strides=2, padding='valid', use_bias=False)(identity)
        identity = BatchNormalization()(identity)
        x += identity
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2))(x)

        # Classification block
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        x = Dense(128,name='fc6')(x)
        x = Activation('relu',name='fc6/relu')(x)
        x = Dense(self.n_classes, name='fc7')(x)
        x = Activation('softmax',name='fc7/softmax')(x)
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input

        model = Model(inputs, x, name='vgg16')
        print(model.summary())
        return model


    def fit(self, x_train, y_train):
        def decay_learning_rate():
            # decay the learning rate each round
            round = min(self.round, 100)
            cosine_decay = 0.5 * (1 + cos(pi * round / 100))
            decayed = (1 - 0.01) * cosine_decay + 0.01
            return 0.001 * decayed
        
        if (self.with_decayed_lr == True):
            lr = decay_learning_rate()
            K.set_value(self.model.optimizer.learning_rate, lr)

        filepath = "results/iemocap/centralized-res11-model-{epoch:02d}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max', period=1)

        if(self.with_adv_training == True and self.round > 1 ):
            half_training_samples = self.x_train.shape[0]//2
            rand_samples_indexes = random.sample(range(x_train.shape[0]), half_training_samples)
            x_train = self.x_train[rand_samples_indexes,]
            y_train = self.y_train[rand_samples_indexes,]
            pre_softmax_layer_name = 'fc7'
            if(self.attack=="deepfool"):
                model= Model(inputs=self.model.model.input, outputs=self.model.model.get_layer(pre_softmax_layer_name).output)
            else:
                model = self.model.model

            classifier = KerasClassifier(model=model)
            adversarial_data = generate_adversarial_data(x_train,y_train,classifier,self.attack)
            x_train= np.concatenate((x_train, adversarial_data), axis=0)
            y_train = np.concatenate((y_train, y_train), axis=0)