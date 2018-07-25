from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM,SimpleRNN,GRU, BatchNormalization,Flatten
from keras.optimizers import Adam

class Rnn:
    def __init__(self,input_dim,output_dim,lr):
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.lr=lr
        self.model=Sequential()
        self.model.add(LSTM(128,input_shape=(1,self.input_dim),return_sequences=True,stateful=False))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(LSTM(128, return_sequences=True, stateful=False))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dense(self.output_dim))
        self.model.add(Flatten())
        self.model.add(Activation('linear'))
        self.model.compile(optimizer=Adam(lr=self.lr),loss='mse',metrics=['mse'])
        self.prob=None

    def predict(self,sample):
        self.prob=self.model.predict(sample)
        return self.prob

    def train_on_batch(self, x, y, batch_size, epoch, valid_x, valid_y):
        return self.model.fit(x, y, batch_size=batch_size, epochs=epoch, validation_data=[valid_x, valid_y])

    def save_model(self,model_path):
        if( model_path is not None and self.model is not None):
            self.model.save_weights(model_path,overwrite=True)

    def load_model(self,model_path):
        if(model_path is not None):
            self.model.load_weights(model_path)
