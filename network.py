
import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
import numpy as np
import os
import math

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
	print(model.input)
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

def reinitialize_final_layers(trained_full_model, trained_conv_model, cutoff_layer):
	untrained_model = VGG16(include_top=False, weights=None)
	mixed_model = Sequential()
	input_layer = trained_conv_model.get_layer("input_1")
	input_shape = input_layer.output_shape[1:3]
	mixed_model.add(input_layer)
	layer_counter = 0
	for layer in untrained_model.layers:
		name = layer.name
		if layer_counter == 0:
			right_layer = untrained_model.get_layer(name)
		elif layer_counter < cutoff_layer - 1:
			right_layer = trained_conv_model.get_layer(name)
		else:
			right_layer = untrained_model.get_layer(name)
		mixed_model.add(right_layer)
		layer_counter += 1
	layer_counter = 0
	for layer in trained_full_model.layers:
		if layer_counter > 0:
			mixed_model.add(layer)
		layer_counter += 1
	mixed_conv_model = Sequential()
	layer_counter = 0
	for layer in mixed_model.layers:
		if layer_counter < cutoff_layer - 1:
			layer.trainable = False
		else: 
			layer.trainable = True
		if layer_counter < 19:
			mixed_conv_model.add(layer)
		layer_counter += 1
	return mixed_model, mixed_conv_model, input_shape




def print_layer_trainable(model):
	""" Prints which layers are trainable and which are not """
	for layer in model.layers:
		print("{0}:\t{1}".format(layer.trainable, layer.name))

def make_generator_train(train_dir, input_shape, batch_size, save_to_dir):
	""" Makes a training data generator from directory """
	datagen_train = ImageDataGenerator(
		rescale=1./255,
		rotation_range=180,
		width_shift_range=0.1,
		height_shift_range=0.1,
		zoom_range=[0.9, 1.5],
		horizontal_flip=True,
		vertical_flip=True,
		fill_mode='reflect',
		brightness_range=[0.9, 1.1]
		)
	generator_train = datagen_train.flow_from_directory(directory=train_dir,
                                                    target_size=input_shape,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    save_to_dir=save_to_dir)
	return generator_train

def make_generator_test(test_dir, input_shape, batch_size):
	""" Makes a data generator for test data from directory """
	datagen_test = ImageDataGenerator(rescale=1./255)
	generator_test = datagen_test.flow_from_directory(directory=test_dir,
                                                  target_size=input_shape,
                                                  batch_size=batch_size,
                                                  shuffle=False)
	return generator_test

def get_conv_part(my_model):
	conv_part = Sequential()
	add = True
	for layer in my_model.layers:
		if add:
			conv_part.add(layer)
		if layer.name == "block5_pool":
			add = False
	return(conv_part)

def create_untrained_model(FC_layers, num_classes):
	""" Creates model. Adds classification layers according to the sizes in list FC_layers """
	model = VGG16(include_top=True, weights='None') #create VGG16
	transfer_layer = model.get_layer('block5_pool') #find final conv_layer
	print(model.input)
	conv_model = Model(inputs=model.input,
	                   outputs=transfer_layer.output) # Isolate convolutional part of VGG16
	new_model = Sequential() # initialize our new model
	new_model.add(conv_model) # add conv part from VGG16
	new_model.add(Flatten()) # add flattening layer
	for size in FC_layers: # Add FC layers according to sizes in FC_layers
		new_model.add(Dense(size, activation='relu'))
	new_model.add(Dense(num_classes, activation='softmax'))
	return new_model
