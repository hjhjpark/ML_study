from keras.models import Sequential
from keras.layers import Activation,Dense,Flatten,BatchNormalization,Dropout
from keras.layers.advanced_activations import PReLU,LeakyReLU
from keras.initializers import TruncatedNormal
from keras.callbacks import EarlyStopping

class Dnn:
    def __init__(self,input_dim,output_dim,optimizer):
        #self.early_stopping=EarlyStopping(monitor='val_loss',patience=10,verbose=1) # early stopping
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.optimizer=optimizer
        self.model=Sequential()
        self.model.add(Dense(128,input_shape=(1,input_dim),activation='sigmoid')) #,kernel_initializer=TruncatedNormal(stddev=0.01)
        #self.model.add(LeakyReLU(alpha=0.01)) #leakyrelu
        #self.model.add(Dropout(0.5)) # dropout
        #self.model.add(BatchNormalization()) # Batch Normalization
        self.model.add(Dense(128,activation='sigmoid'))
        # self.model.add(Dropout(0.5)) # dropout
        # self.model.add(BatchNormalization()) # Batch Normalization
        self.model.add(Dense(128,activation='sigmoid'))
        # self.model.add(Dropout(0.5)) # dropout
        # self.model.add(BatchNormalization()) # Batch Normalization
        self.model.add(Dense(output_dim))
        self.model.add(Flatten())
        self.model.add(Activation('softmax'))
        self.model.compile(optimizer=self.optimizer,loss='categorical_crossentropy',metrics=['accuracy','categorical_crossentropy'])
        self.prob=None

    def predict(self,sample):
        self.prob=self.model.predict(sample)[0]
        return self.prob

    def train_on_batch(self,x,y,batch_size,epoch,valid_x,valid_y):
        return self.model.fit(x,y,batch_size=batch_size,epochs=epoch,validation_data=[valid_x,valid_y]) #,callbacks=[self.early_stopping]

    def save_model(self,model_path):
        if( model_path is not None and self.model is not None):
            self.model.save_weights(model_path,overwrite=True)

    def load_model(self,model_path):
        if(model_path is not None):
            self.model.load_weights(model_path)



