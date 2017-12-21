# Road segmentation pipeline.
# Dependencies: keras + tensorflow backend, opencv-python, imutils
import itertools
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import losses
from keras.datasets import mnist
from keras.models import Sequential, model_from_json

from keras.layers import Dense, Dropout, Flatten, Reshape, Conv2D, MaxPooling2D, LeakyReLU
from keras import backend as K
from keras.optimizers import Adam

#from keras.utils.training_utils import multi_gpu_model

from helpers import *

patch_size = 16 # each patch is 16*16 pixels

imgs, gt_imgs = training_dataset(limit=5) # Memory hungry, use e.g. 10 for testing

train_data = [
    (patch, gt_patch)
    for i in range(len(imgs))
    for patch, gt_patch in itertools.islice(gen_random_patches(reflect_border(imgs[i], patch_size, 2),
                                                              reflect_border(gt_imgs[i], patch_size, 2), i), 25*25*8)]


X = np.asarray([img for img, _ in train_data])
y = np.asarray([value_to_class(gt_patch) for _, gt_patch in train_data]).reshape((-1, 1))

# Model parameters

batch_size = 64 # * 8 for multi-gpu
num_filters_1 = 16
num_filters_2 = 32
num_filters_3 = 64
num_filters_4 = 128

epochs = 1

def build_model(gpus=0):
    model = Sequential()
    model.add(Conv2D(num_filters_1, kernel_size=(4, 4), input_shape=(80, 80, 3)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(num_filters_2, kernel_size=(4,4)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(num_filters_3, kernel_size=(4,4)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(num_filters_4, kernel_size=(4,4)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    simple_model = model
    if gpus > 1:
        model = multi_gpu_model(simple_model, gpus=gpus)

    model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=1e-3),
              metrics=['accuracy'])
    
    return (model, simple_model)

model, simple_model = build_model(0)
model.summary()

model.fit(X, y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          shuffle=True)

# save the model and the weights to files
model_json = simple_model.to_json()
with open('model_18', 'w') as f:
    f.write(model_json)

simple_model.save_weights('weights_18')

# Helper variables and functions for the submission
foreground_threshold = 0.5 # percentage of pixels > 1 required to assign a foreground label to a patch
patch_size = 16

def pred_print(img, img_patches):
    pred = [
        np.round((np.median(model.predict(np.asarray(image_trans(patch))))))
        for patch in img_patches
    ]
    pred = np.asarray(pred)
    disp_img_pred(img, pred, patch_size)
    return pred


# Creates submission file using a trained model and given images
def submission_with_model(model, submissionfilename):

    root_dir = "data/test_set_images/"

    # Get filenames and images for all the 50 submission images
    image_dir = [root_dir + "test_{}/".format(i) for i in range(1, 51)]
    filenames = [fn for imdir in image_dir for fn in os.listdir(imdir)]
    images = [load_image(image_dir[i-1] + filenames[i-1]) for i in range(1, 51)]

    # Reflect borders
    im_borders = [reflect_border(im, patch_size, 2) for im in images]
    
    # Get 16x16 patches of the images
    imgs_patched = [image_to_inputs(im, patch_size) for im in im_borders]
    
    # Use model to predict, predictions are labels in a list of labels per image
    predictions = []
    for i in range(1,51):
        predictions.append(pred_print(images[i-1], np.asarray(imgs_patched[i-1])))
    
    with open(submissionfilename, 'w') as f:
        f.write('id,prediction\n')
        for nr in range(1, 51):
            f.writelines(
                '{}\n'.format(s)
                for s in img_to_submission_strings(predictions[nr-1],
                                                   nr,
                                                   images[nr-1].shape[1],
                                                   images[nr-1].shape[0],
                                                   patch_size))
            
# Use the model to predict and create the submission file.
submission_with_model(model, 'submission_test.csv')