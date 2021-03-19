from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

entries=10000
x=np.random.randn(entries,1)*np.pi
y=np.sin(x)+(np.random.randn(entries,1)*.1)  #adding som noise into the process
#xReal=np.linspace(-np.pi*3,np.pi*3,entries/100)
#xReal=np.transpose(np.array([xReal]))
#yReal=np.sin(xReal) 
xTrain=x[0:int(np.round(entries*.85)),0]
yTrain=y[0:int(np.round(entries*.85)),0]
x_test=x[int(np.round(entries*.85)):entries,0]
y_test=y[int(np.round(entries*.85)):entries,0]

def load_model():
    model = Sequential()
    model.add(Dense(units=32,activation='relu',input_dim=1))
    model.add(Dense(units=32,activation='relu'))
    model.add(Dense(units=32,activation='relu'))
    model.add(Dense(units=32,activation='relu'))
    model.add(Dense(units=1,activation='tanh'))
    model.summary()
    model.compile(loss='mse',optimizer='Adam')
    return model

model = load_model()
callbacks = [EarlyStopping(monitor='val_loss', patience=1),
             ModelCheckpoint(filepath='simpleModel.h5', monitor='val_loss', save_best_only=True)] 
#training my model
model.fit(xTrain,yTrain,epochs=10,callbacks=callbacks,batch_size=10000,validation_data=(x_test, y_test))
