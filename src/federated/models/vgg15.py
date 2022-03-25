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

class VGG15(FL_MODEL):
    def compile(self, input_shape = (None,None,1) ):
        img_input = Input(shape=input_shape)
        # Block 1
        x = Conv2D(64, (3, 3), padding='same', name='conv1_1')(img_input)
        x = BatchNormalization(name='conv1_1_bn')(x)
        x = Activation('relu', name='conv1_1_relu')(x)
        x = Conv2D(64, (3, 3), padding='same', name='conv1_2')(x)
        x = BatchNormalization(name='conv1_2_bn')(x)
        x = Activation('relu', name='conv1_2_relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)
        # Block 2
        x = Conv2D(128, (3, 3), padding='same', name='conv2_1')(x)
        x = BatchNormalization(name='conv2_1_bn')(x)
        x = Activation('relu', name='conv2_1_relu')(x)
        x = Conv2D(128, (3, 3), padding='same', name='conv2_2')(x)
        x = BatchNormalization(name='conv2_2_bn')(x)
        x = Activation('relu', name='conv2_2_relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)
        # Block 3
        x = Conv2D(256, (3, 3), padding='same', name='conv3_1')(x)
        x = BatchNormalization(name='conv3_1_bn')(x)
        x = Activation('relu', name='conv3_1_relu')(x)
        x = Conv2D(256, (3, 3), padding='same', name='conv3_2')(x)
        x = BatchNormalization(name='conv3_2_bn')(x)
        x = Activation('relu', name='conv3_2_relu')(x)
        x = Conv2D(256, (3, 3), padding='same', name='conv3_3')(x)
        x = BatchNormalization(name='conv3_3_bn')(x)
        x = Activation('relu', name='conv3_3_relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)
        # Block 4
        x = Conv2D(512, (3, 3), padding='same', name='conv4_1')(x)
        x = BatchNormalization(name='conv4_1_bn')(x)
        x = Activation('relu', name='conv4_1_relu')(x)
        x = Conv2D(512, (3, 3), padding='same', name='conv4_2')(x)
        x = BatchNormalization(name='conv4_2_bn')(x)
        x = Activation('relu', name='conv4_2_relu')(x)
        x = Conv2D(512, (3, 3), padding='same', name='conv4_3')(x)
        x = BatchNormalization(name='conv4_3_bn')(x)
        x = Activation('relu', name='conv4_3_relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)
        # Block 5
        x = Conv2D(512, (3, 3), padding='same', name='conv5_1')(x)
        x = BatchNormalization(name='conv5_1_bn')(x)
        x = Activation('relu', name='conv5_1_relu')(x)
        x = Conv2D(512, (3, 3), padding='same', name='conv5_2')(x)
        x = BatchNormalization(name='conv5_2_bn')(x)
        x = Activation('relu', name='conv5_2_relu')(x)
        x = Conv2D(512, (3, 3), padding='same', name='conv5_3')(x)
        x = BatchNormalization(name='conv5_3_bn')(x)
        x = Activation('relu', name='conv5_3_relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)
        # Classification block
        x = GlobalAveragePooling2D(name='pool6')(x)
        x = Dense(256, name='fc6')(x)
        x = Activation('relu', name='fc6/relu')(x)
        x = Dense(self.n_classes, name='fc7')(x)
        x = Activation('softmax', name='fc7/softmax')(x)
        model = Model(img_input, x, name='vgg16')  
        # Optimiser
        decay_steps = 20000
        lr_decayed_fn = CustomCosineDecay(0.001, decay_steps)
        optimizer = tf.keras.optimizers.Adam(lr_decayed_fn)
        opt = SGD(lr=1e-2, momentum=0.9, decay=1e-2/self.round)
        # Compile
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.model=model

    def fit(self,x_train,y_train):
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


        history = self.model.fit(x_train, y_train, batch_size=self.batch_size ,epochs=self.epochs, validation_split=0.05, verbose=1)
        return history
