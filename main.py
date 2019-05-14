from csv_reader import get_labels
from csv_reader import sort_data

from network import *
from sklearn.utils.class_weight import compute_class_weight

labels = get_labels("Data_Osteo_Tiles/ML_Features_1144.csv")
label_counter = [0]*3

num_classes = 3 
FC_layers = [1024, 256] #layer sizes of FC layers in classification part
cutoff_layer = 20
batch_size  = 20
epochs = 20
steps_per_epoch = 100
train_dir = "Data_Osteo_Tiles/train_data"
test_dir = "Data_Osteo_Tiles/test_data"
save_to_dir = "Data_Osteo_Tiles/save_to_directory"
input_shape = []



for label in labels.values():
	label_counter[label] += 1

print("label percentages: ")
for counter in label_counter:
	print(counter/len(labels))

#sort_data(labels)

""" Create model & compile """
new_model, conv_model = create_model(FC_layers, num_classes)
optimizer = Adam(lr=1e-5)
loss = 'categorical_crossentropy'
metrics = ['categorical_accuracy']
conv_model = set_layers_trainable(conv_model, cutoff_layer)
print_layer_trainable(conv_model)
new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

input_shape = conv_model.layers[0].output_shape[1:3]
print(input_shape)

"""create data generators"""
generator_train = make_generator_train(train_dir, input_shape, batch_size, save_to_dir)
cls_train = generator_train.classes
generator_test = make_generator_test(test_dir, input_shape, batch_size)
steps_test = generator_test.n / batch_size
cls_test = generator_test.classes

class_weight = compute_class_weight(class_weight='balanced',
                                    classes=np.unique(cls_train),
                                    y=cls_train)

"""Train model"""
history = new_model.fit_generator(generator=generator_train,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  class_weight=class_weight,
                                  validation_data=generator_test,
                                  validation_steps=steps_test)

