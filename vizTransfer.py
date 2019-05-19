
import numpy as np
from PIL import Image as pil_image
from tensorflow.python.keras.preprocessing.image import save_img
from tensorflow.python.keras import layers
from tensorflow.python.keras.applications import vgg16
from tensorflow.python.keras import backend as K
import network as nt
import tensorflow as tf

#This function Creates an image with random pixel values and then uses
#gradient descent to find the image that yields the maximum Average
#activation values. It also uses an upscaling method in order to increase
# the pattern frequency.
def compute_gradients(output_dim,model_input,layer_output,step_size,epochs,filter_index):
    number_of_upscales = 9;
    scaling_factor = 1.2;
    # the initial image dimensions,
    image_dimensions = [int(output_dim / (scaling_factor ** number_of_upscales)),
        int(output_dim / (scaling_factor ** number_of_upscales))]

    #creating an image with random pixel values in range [0,1] and creating a
    #keras function that computes the normalized gradients
    random_img = np.random.uniform(0,255,(1, image_dimensions[0], image_dimensions[1], 3))/255
    loss_val = -K.mean(layer_output[:, :, :, filter_index])
    gradients = K.gradients(loss_val, model_input)[0]
    gradients = gradients/(K.sqrt(K.mean(K.square(gradients))) + K.epsilon())
    grad_func = K.function([model_input], [loss_val, gradients])
    #Each time image is upsclaed a gradient descent is made.
    for up in reversed(range(number_of_upscales)):
        for i in range(epochs):
            loss_value, grads_value = grad_func([random_img])
            random_img -= grads_value * step_size
        image_dimensions = [int(output_dim / (scaling_factor ** up)),
            int(output_dim / (scaling_factor ** up))]

        #After gradient descent our image might no longer
        #be in the correct range [0,1] or [0,255]. We need to
        #clip the values.
        filter_img = random_img[0]*255
        filter_img = np.clip(filter_img,0,255).astype('uint8')
        #resize and reshape according to upscaled image.
        filter_img = np.array(pil_image.fromarray(filter_img).resize(image_dimensions,
                                                                   pil_image.BICUBIC))
        random_img = (filter_img/255).reshape(1, filter_img.shape[0],
            filter_img.shape[1], filter_img.shape[2])
    filter_img = random_img[0]*255
    return filter_img


def draw_image(filter,layer):
    save_img('vgg_test_{0:}_{1:}x{1:}.png'.format(layer, 1), filter)

def visualize_layer(model, layer):
    model_input = model.inputs[0]
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    output_layer = layer_dict[layer]
    layer_output = output_layer.output
    step_size=1.
    epochs=15
    #upscaling_steps=9
    #upscaling_factor=1.2
    output_dim=412
    filter_index = 0

    filter = compute_gradients(output_dim,model_input,layer_output,step_size,epochs,filter_index)
    draw_image(filter,layer)
