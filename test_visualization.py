from network import *
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf


# load the picture to use
file_path = "Data_Osteo_Tiles/all_data/class_1/Case 3 A12-8900-9585.jpg"
img = Image.open(file_path)
new_size = 224, 224
img = img.resize(new_size)
print('layer and image ready')

visualize_layer(img)
print('layer plotted')