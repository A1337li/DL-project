import csv
import numpy as np
import os
import random
from PIL import Image

"""'Non-Tumor' = Non-Tumor -> 0, 
'Non-Viable-Tumor' = Necrosis -> 1,
'Viable' = 'viable: non-viable' = Viable -> 2 """
text_labels = ['Non-Tumor', 'Non-Viable-Tumor', 'Viable', 'viable: non-viable']
# labels = []

def get_labels(file_name):
    """ Takes the file name of a csv-file (the big one), 
        and outputs a dictionary {image_file_name: label} """
    with open(file_name, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        files_and_labels = {}
        for row in csv_reader:
            text_label = row["classification"]
            title = row["image.name"]
            image_file_name = title + ".jpg"
            if not text_label in text_labels:
                text_labels.append (text_label)
            label = text_labels.index(text_label)
            if label == 3:
                label = 2
            files_and_labels[image_file_name] = label
    return(files_and_labels)

def sort_data(dictionary): 
    folder_path = "Data_Osteo_Tiles/all_data/"
    for key in dictionary: 
        c = dictionary[key]
        file_path = folder_path + key
        print(file_path)
        img = Image.open(file_path)
        save_folder_path = "Data_Osteo_Tiles/all_data/class_" + str(c) + "/" + key
        print(save_folder_path)
        img.save(save_folder_path, "JPEG")

def separate_testdata(dictionary):
    folder_path_0 = "Data_Osteo_Tiles/all_data/class_0/"
    folder_path_1 = "Data_Osteo_Tiles/all_data/class_1/"
    folder_path_2 = "Data_Osteo_Tiles/all_data/class_2/"
    folder_paths = [folder_path_0, folder_path_1, folder_path_2]
    save_path_0 = "Data_Osteo_Tiles/test_data/test_class_0/"
    save_path_1 = "Data_Osteo_Tiles/test_data/test_class_1/"
    save_path_2 = "Data_Osteo_Tiles/test_data/test_class_2/"
    save_paths = [save_path_0, save_path_1, save_path_2]
    save_train_path_0 = "Data_Osteo_Tiles/train_data/train_class_0/"
    save_train_path_1 = "Data_Osteo_Tiles/train_data/train_class_1/"
    save_train_path_2 = "Data_Osteo_Tiles/train_data/train_class_2/"
    save_train_paths = [save_train_path_0, save_train_path_1, save_train_path_2]
    fraction_test = 0.3
    fraction_train = 1 - fraction_test
    n_data = len(dictionary.keys())
    n_train = int(round(n_data * fraction_train))
    n_test = n_data - n_train
    counter = 0
    while counter < n_test:
        rand_key = random.choice(list(dictionary.keys()))
        cat = dictionary[rand_key]
        print(cat)
        dictionary.pop(rand_key)
        file_path = folder_paths[cat] + rand_key
        img = Image.open(file_path)
        save_path = save_paths[cat] + rand_key
        print(save_path)
        img.save(save_path, "JPEG")
        counter += 1
    for key in dictionary.keys():
        cat = dictionary[key]
        file_path = folder_paths[cat] + key
        img = Image.open(file_path)
        save_path = save_train_paths[cat] + key
        img.save(save_path, "JPEG")


# csv_file_path = "Data_Osteo_Tiles/ML_Features_1144.csv"
# dictionary = get_labels(csv_file_path)
# separate_testdata(dictionary)







