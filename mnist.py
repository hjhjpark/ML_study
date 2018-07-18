from DNN import Dnn
import numpy as np
import csv
import random
from keras.optimizers import sgd,adagrad,rmsprop,adadelta,adam
import matplotlib.pyplot as plt


batch_size=64
input_dim=784
output_dim=10
learning_rate=0.001
train_size=0.8
num_epoch=100

f=open("./train.csv",'r',encoding='utf-8',newline='')
reader=csv.reader(f,delimiter=',')

raw_data=[[int(e) for e in r] for r in list(reader)[1:]]
random.shuffle(raw_data)
data_set=np.array(raw_data)
train_x=data_set[:len(data_set)*train_size,1:]
train_y=np.eye(10)[data_set[:len(data_set)*train_size,0]]
valid_x=data_set[len(data_set)*train_size:,1:]
valid_y=np.eye(10)[data_set[len(data_set)*train_size:,0]]
plt.xlabel("epoch")
plt.ylabel("loss")

# stochastic gradient descent
dnn=Dnn(input_dim,output_dim,sgd(lr=learning_rate))
dnn.model.summary()
hist=dnn.train_on_batch(np.array(train_x).reshape(-1,1,input_dim),train_y,batch_size,num_epoch,np.array(valid_x).reshape(-1,1,input_dim),valid_y)
plt.plot(range(len(hist.history['val_loss'])),hist.history['val_loss'],label="SGD")

# momentum
dnn=Dnn(input_dim,output_dim,sgd(lr=learning_rate,momentum=0.9))
dnn.model.summary()
hist=dnn.train_on_batch(np.array(train_x).reshape(-1,1,input_dim),train_y,batch_size,num_epoch,np.array(valid_x).reshape(-1,1,input_dim),valid_y)
plt.plot(range(len(hist.history['val_loss'])),hist.history['val_loss'],label="MOMENTUM")

# nestrov accelerated gradient
dnn=Dnn(input_dim,output_dim,sgd(lr=learning_rate,momentum=0.9,nesterov=True))
dnn.model.summary()
hist=dnn.train_on_batch(np.array(train_x).reshape(-1,1,input_dim),train_y,batch_size,num_epoch,np.array(valid_x).reshape(-1,1,input_dim),valid_y)
plt.plot(range(len(hist.history['val_loss'])),hist.history['val_loss'],label="NAG")

# adagrad
dnn=Dnn(input_dim,output_dim,adagrad(lr=learning_rate))
dnn.model.summary()
hist=dnn.train_on_batch(np.array(train_x).reshape(-1,1,input_dim),train_y,batch_size,num_epoch,np.array(valid_x).reshape(-1,1,input_dim),valid_y)
plt.plot(range(len(hist.history['val_loss'])),hist.history['val_loss'],label="ADAGRAD")

#rmsprop
dnn=Dnn(input_dim,output_dim,rmsprop(lr=learning_rate))
dnn.model.summary()
hist=dnn.train_on_batch(np.array(train_x).reshape(-1,1,input_dim),train_y,batch_size,num_epoch,np.array(valid_x).reshape(-1,1,input_dim),valid_y)
plt.plot(range(len(hist.history['val_loss'])),hist.history['val_loss'],label="RMSPROP")

#adam
dnn=Dnn(input_dim,output_dim,adam(lr=learning_rate))
dnn.model.summary()
hist=dnn.train_on_batch(np.array(train_x).reshape(-1,1,input_dim),train_y,batch_size,num_epoch,np.array(valid_x).reshape(-1,1,input_dim),valid_y)
plt.plot(range(len(hist.history['val_loss'])),hist.history['val_loss'],label="ADAM")

plt.legend(bbox_to_anchor=(1,1),loc=6,borderaxespad=0.)
plt.show()

# test
'''
f=open("./test.csv",'r',encoding='utf-8',newline='')
reader=csv.reader(f,delimiter=',')
raw_data=[[int(e) for e in r] for r in list(reader)[1:]]
data_set=np.array(raw_data)
test_x=data_set[:,:].reshape(-1,1,input_dim)
for i in range(len(test_x)):
    print(np.argmax(dnn.predict(test_x[i:i+1])))
'''

