from keras.models import Sequential
from keras.layers import LSTM,Dense, RepeatVector
from model import FL_MODEL


class Autoencoder(FL_MODEL):
        
    def compile(self):
        model = Sequential()
        #TODO replace the hardcoded input layer size by config value
        model.add(LSTM(64, activation = 'relu', input_shape=(32, 32),return_sequences = True))
        model.add(LSTM(12, activation = 'relu', return_sequences = False))
        model.add(RepeatVector(16))
        #TODO replace the hardcoded output layer size by config value
        model.add(LSTM(12, activation = 'relu', return_sequences = True))
        model.add(LSTM(64, activation = 'relu', return_sequences=True))
        # one hot encoded vector (1: normal, 0:anomaly)
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='mse', optimizer='adam')
        self.model=model
        
    
    def fit(self,x_train,y_train):
        #TODO replace static values by config values
        history = self.model.fit(x_train, y_train)
        return history
    
        