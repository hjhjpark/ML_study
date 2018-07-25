import numpy as np
import csv
import matplotlib.pyplot as plt
from RNN import Rnn

seq_length=10
num_epoch=100
batch_size=64
train_size=0.8
val_size=0.1
test_size=0.1

# data preprocessing
fr=open("./005930.csv","r",encoding='utf-8',newline='')
reader=csv.reader(fr,delimiter=',')
data=np.array([float(row[4]) for row in list(reader)[1:]])[73:]
data_min=np.min(data)
data_max=np.max(data)
data=(data-data_min)/(data_max-data_min) # min max normalization
data_x=[]
data_y=[]
for i in range(len(data)-seq_length):
    data_x.append(data[i:i+seq_length])
    data_y.append(data[i+seq_length])
X=np.array(data_x).reshape(len(data_x),1,-1)
Y=np.array(data_y).reshape(len(data_y),1)

train_val_x,train_val_y=X[:int((train_size+val_size)*len(data_x))],Y[:int((train_size+val_size)*len(data_y))]
train_x,train_y=train_val_x[:int((train_size)*len(data_x))],train_val_y[:int((train_size)*len(data_y))]
valid_x,valid_y=train_val_x[int((train_size)*len(data_x)):],train_val_y[int((train_size)*len(data_y)):]
test_x,test_y=X[int((train_size+val_size)*len(data_x)):],Y[int((train_size+val_size)*len(data_y)):]
network=Rnn(seq_length,1,0.001)
hist=network.train_on_batch(train_x,train_y,batch_size,num_epoch,valid_x,valid_y)
test_predict=network.predict(test_x)

# visualization
plt.plot(test_predict*(data_max-data_min)+data_min,label='predict')
plt.plot(test_y*(data_max-data_min)+data_min,label='true')
plt.legend(bbox_to_anchor=(1,1),loc=2,borderaxespad=0.)
plt.show()
plt.close()

plt.plot(hist.history['val_loss'],color='g',label='loss')
plt.legend(bbox_to_anchor=(1,1),loc=1,borderaxespad=0.)
plt.show()