from keras.activations import relu
from keras.layers.merge import add
import tensorflow as tf
from tensorflow import keras 
from keras import Sequential 
from keras.layers import Dense 
from keras.datasets import mnist 
from keras.layers import Conv2D,MaxPool2D,Flatten 
(x_train,y_train1),(x_test,y_test1) =mnist.load_data()
y_train=keras.utils.to_categorical(y_train1,10)
y_test =keras.utils.to_categorical(y_test1,10)
x_train=x_train.astype('float32')/255 
x_test=x_test.astype('float32')/255 
model =Sequential() 
model.add(Conv2D(32,5,strides=(1,1),padding='same',activation=tf.nn.relu,input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2,2),strides=(1,1),padding='valid'))
model.add(Conv2D(64,5,strides=(1,1),padding='same',activation=tf.nn.relu,input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2,2),strides=(1,1),padding='valid'))
model.add(Flatten()) 
model.add(Dense(128,activation=tf.nn.relu))
model.add(Dense(10,activation=tf.nn.softmax))
model.summary()
x_train=x_train.reshape(60000,28,28,1)
x_test=x_test.reshape(10000,28,28,1)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x=x_train,y=y_train,epochs=5,batch_size=128)
test_loss,test_accurrcy=model.evaluate(x_test,y_test)
model.save("HandWrriten_Digit_image recognition.h5")