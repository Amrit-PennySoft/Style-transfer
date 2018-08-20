import numpy as numpy
from keras.applications import vgg16
from keras import backend as K 
from keras.preprocessing.image import load_img, img_to_array
#helper class
from PIL import Image as img﻿
import image_processing as img_to_array
#image display
from IPython.display import image

#get tensor representations of images
bse_image = K.variable(img.preprocess_image('./base_image.jpg'))
style_reference_image = K.variable(img.preprocess_image('./style_image.jpg'))
combination_image = K.placeholder((1,400,711,3))

image('./style_image.jpg')

#combining the 3 images into a single keras tensor
input_tensor = L.concatennate([base_image,
                               style_reference_image,
                               combination_image], axis=0)

#build with VGG network with the 3 images
model = vgg16.VGG16(input_tensor=input_tensor,
                    weights='imagenet', include_top=False)

print('model loaded')

#combine loss functions into a isngle scalar
loss =img.combination_loss(model, combination_image)
print (loss)

#get the gradients of the generated image for the loss
grads = K.gradients(loss, combination_image)
print (grads)

#run optimization (L-BFGS) over the pixels of the generated image to minimie loss
combination_image = img.minimize_loss(grads, loss, combination_image)

image(combination_image)

