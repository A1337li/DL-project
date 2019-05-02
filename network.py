
import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
import numpy as np
import os

from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam, RMSprop

def create_model(FC_layers, num_classes):
	""" Creates model. Adds classification layers according to the sizes in list FC_layers """
	model = VGG16(include_top=True, weights='imagenet') #create VGG16
	transfer_layer = model.get_layer('block5_pool') #find final conv_layer
	conv_model = Model(inputs=model.input,
	                   outputs=transfer_layer.output) # Isolate convolutional part of VGG16
	new_model = Sequential() # initialize our new model
	new_model.add(conv_model) # add conv part from VGG16
	new_model.add(Flatten()) # add flattening layer
	for size in FC_layers: # Add FC layers according to sizes in FC_layers
		new_model.add(Dense(size, activation='relu'))
	new_model.add(Dense(num_classes, activation='softmax'))
	return new_model, conv_model

def set_layers_trainable(conv_model, cutoff_layer):
	""" Sets the layers in conv_model before cutoff_layer non-trainable, and the rest trainable """
	for i in range(len(conv_model.layers)):
		layer = conv_model.layers[i]
		if i < cutoff_layer - 1:
			layer.trainable = False
		else:
			layer.trainable = True
	return conv_model

def print_layer_trainable(model):
	""" Prints which layers are trainable and which are not """
	for layer in model.layers:
		print("{0}:\t{1}".format(layer.trainable, layer.name))






