import tensorflow as tf 

from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D

def create_our_model(): 

	# BLOCK 1
	model = Sequential()
	model.add(Conv2D(64, (3, 3), input_shape = [224, 224, 3]))
		# arg1 = n_filters, arg2 = filter size, arg3 = input size for making it dynamic 
	model.add(Conv2D(64, (3, 3)))
	#model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size = (2, 2)))
	
	# BLOCK 2
	model.add(Conv2D(128, (3, 3)))
	model.add(Conv2D(128, (3, 3)))
	#model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size = (2, 2)))

	# BLOCK 3
	model.add(Conv2D(256, (3, 3)))
	model.add(Conv2D(256, (3, 3)))
	model.add(Conv2D(256, (3, 3)))
	#model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size = (2, 2)))

	model.add(Flatten())
	model.add(Dense(256, activation = 'relu'))
	#model.add(Dense(128, activation = 'relu'))
	model.add(Dense(3, activation='softmax')) # 3 stands for number of classes

	return model

