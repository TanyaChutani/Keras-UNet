import tensorflow as tf
import cv2
import os
import tensorflow as tf
from keras.engine.input_layer import Input
from keras.layers import GRU,Conv2D,MaxPooling2D,Conv2DTranspose,Dropout,concatenate,Activation
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import pdbs

#Model Unet
def contraction_block(inputs,nfilter):
  conv = Conv2D(filters=nfilter,kernel_size= (3,3), activation = 'relu', padding = 'same')(inputs)
  conv = Conv2D(filters=nfilter, kernel_size=(3,3), activation = 'relu', padding = 'same')(conv)
  pool = MaxPooling2D(pool_size=(2, 2))(conv)
  return conv,pool

def expansion_block(inputs,nfilter,residual):
  transpose_conv=Conv2DTranspose(filters=nfilter,kernel_size=(3,3),strides=(2,2),padding='same')(inputs)
  skip_connect=concatenate([transpose_conv,residual],axis=3)
  conv=Conv2D(filters=nfilter,kernel_size=(3,3),activation='relu',padding='same')(skip_connect)
  conv=Conv2D(filters=nfilter,kernel_size=(3,3),activation='relu',padding='same')(conv)
  return conv

def unet_model():
  height=256
  width=256
  nclasses=13
  filters=64
  input_layer = Input(shape=(height,width,3))
  print(input_layer.shape)
  #Contraction path 4 block
  conv1,output1=contraction_block(input_layer,nfilter=filters*1)
  print(conv1.shape)
  conv2,output2=contraction_block(output1,nfilter=filters*2)
  print(conv2.shape)
  conv3,output3=contraction_block(output2,nfilter=filters*4)
  print(conv3.shape)
  conv4,output4=contraction_block(output3,nfilter=filters*8)
  conv4=Dropout(0.5)(conv4)
  print(conv4.shape)

  conv5,output=contraction_block(output4,nfilter=filters*16)
  print(conv5.shape)

  #Expansion path 4 block
  output6=expansion_block(inputs=conv5,nfilter=filters*8,residual=conv4)
  print(output6.shape)
  output7=expansion_block(inputs=output6,nfilter=filters*4,residual=conv3)
  print(output7.shape)
  output8=expansion_block(inputs=output7,nfilter=filters*2,residual=conv2)
  print(output8.shape)
  output9=expansion_block(inputs=output8,nfilter=filters*1,residual=conv1)
  print(output9.shape)


  #Output
  output_layer = Conv2D(filters=nclasses, kernel_size=(1, 1),activation='softmax')(output9)
  print(output_layer.shape)
  model=Model(inputs=input_layer,outputs=output_layer)
  model.compile(optimizer=Adam(1e-4),loss='categorical_crossentropy',metrics=['accuracy'])
  model.summary()
  return model
model=unet_model()
