import csv
import numpy as np

text_labels = ['Non-Tumor', 'Non-Viable-Tumor', 'Viable', 'viable: non-viable']
labels = []

def get_labels(file_name):
    with open(file_name, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        text_labels = []
        labels = []
        for row in csv_reader:
            text_label = row["classification"]
            if not text_label in text_labels:
                text_labels.append (text_label)
            label = text_labels.index(text_label)
            labels.append(label)
    return(labels)
