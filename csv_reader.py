import csv
import numpy as np

"""'Non-Tumor' = Non-Tumor -> 0, 
'Non-Viable-Tumor' = Necrosis -> 1,
'Viable' = 'viable: non-viable' = Viable -> 2 """
text_labels = ['Non-Tumor', 'Non-Viable-Tumor', 'Viable', 'viable: non-viable']
labels = []

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
