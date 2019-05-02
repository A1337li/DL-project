from csv_reader import get_labels
<<<<<<< HEAD
from csv_reader import sort_data

=======
from network import *
>>>>>>> 6ba838041405d7128ae331473127364cd9f38a2d

labels = get_labels("Data_Osteo_Tiles/ML_Features_1144.csv")
label_counter = [0]*3

num_classes = 3 
FC_layers = [1024, 256] #layer sizes of FC layers in classification part
cutoff_layer = 10
epochs = 20
steps_per_epoch = 100

for label in labels.values():
	label_counter[label] += 1

print("label percentages: ")
for counter in label_counter:
	print(counter/len(labels))

<<<<<<< HEAD
sort_data(labels)
=======

""" Create model & compile """
new_model, conv_model = create_model(FC_layers, num_classes)
optimizer = Adam(lr=1e-5)
loss = 'categorical_crossentropy'
metrics = ['categorical_accuracy']
conv_model = set_layers_trainable(conv_model, cutoff_layer)
print_layer_trainable(conv_model)
new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

""" Need function that reads in data and then we can train the model using 
	history = new_model.fit_generator(...)"""
>>>>>>> 6ba838041405d7128ae331473127364cd9f38a2d
