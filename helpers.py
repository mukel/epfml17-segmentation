# Helper functions for loading and manipulating the data.
import os,sys
import re
import cv2
import imutils
from PIL import Image
import matplotlib.image as mpimg
import numpy as np
import random


def load_image(infilename):
    data = mpimg.imread(infilename)
    return data


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[i:i+w, j:j+h]
            else:
                im_patch = im[i:i+w, j:j+h, :]
            list_patches.append(im_patch)
    return list_patches


def reflect_border(im, patch_size, count):
    bordersize = patch_size*count
    border = cv2.copyMakeBorder(im,
                              top=bordersize,
                              bottom=bordersize, 
                              left=bordersize,
                              right=bordersize,
                              borderType=cv2.BORDER_REFLECT)
    return border

def rotate_image(img, rot_degrees=[0, 90, 180, 270]):
    """
    Returns a list with the rotated images.
    """
    return [
        imutils.rotate_bound(img, rot)
        for rot in rot_degrees
    ]


def flip_image(img):
    return cv2.flip(img, 1)


def image_trans(img):
    return [
        t_img
        for rot_img in rotate_image(img)
        for t_img in [rot_img, flip_image(rot_img)]
    ]


def training_dataset(limit=100, root_dir='./data/training/'):
    image_dir = root_dir + "images/"
    files = os.listdir(image_dir)
    n = min(limit, len(files))
    print('Loading', n, 'training images')
    imgs = [load_image(image_dir + files[i]) for i in range(n)]
    gt_dir = root_dir + "groundtruth/"
    print('Loading', n, 'groundtruth images')
    gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]
    return (imgs, gt_imgs)

# Convert array of labels to an image

def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[i:i+w, j:j+h] = labels[idx]
            idx = idx + 1
    return im

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def print_prediction(img, gt_img, model):
    img_patches = [
        img_patch
        for img_patch in image_to_patches(reflect_border(img, patch_size, 2), get_image_crop)
    ]

    gt_labeled_patches = [
        value_to_class(gt_patch)
        for gt_patch in image_to_patches(reflect_border(gt_img, patch_size, 2), get_groundtruth_crop)
    ]

    predicted_patches = model.predict(np.asarray(img_patches))
    
    w = img.shape[0]
    h = img.shape[1]
    predicted_im = label_to_img(w, h, patch_size, patch_size, predicted_patches)
    
    
    print("Ground truth")
    cimg = concatenate_images(img, gt_img)
    fig1 = plt.figure(figsize=(10, 10)) # create a figure with the default size 
    plt.imshow(cimg, cmap='Greys_r')
    plt.show()
    
    print("Ground truth patches")
    gt_im_patches = label_to_img(w, h, patch_size, patch_size, gt_labeled_patches)

    cimg = concatenate_images(img, gt_im_patches)
    fig1 = plt.figure(figsize=(10, 10)) # create a figure with the default size 
    plt.imshow(cimg, cmap='Greys_r')
    plt.show()
    
    print("Predicted patches")

    cimg = concatenate_images(img, predicted_im)
    fig1 = plt.figure(figsize=(10, 10)) # create a figure with the default size 
    plt.imshow(cimg, cmap='Greys_r')
    plt.show()
    
    print("Prediction overlay")

    new_img = make_img_overlay(img, predicted_im)

    plt.imshow(new_img)

def image_to_patches(img, func):
    p = img.shape
    assert p[0] == p[1]
    n = p[0]
    patches = [
        func(img, i, j)
        for i in range(2*patch_size, n - 3*patch_size+1, patch_size)
        for j in range(2*patch_size, n - 3*patch_size+1, patch_size)
    ]
    return patches

def get_image_crop(img, i, j):
    return img[i-2*patch_size:i+3*patch_size, j-2*patch_size:j+3*patch_size, :]

def get_groundtruth_crop(gt_img, i, j):
    return gt_img[i:i+patch_size, j:j+patch_size]

def value_to_class(v, foreground_threshold = 0.25):
    df = np.mean(v)
    return (df > foreground_threshold) * 1


def gen_random_patches(img, gt, seed=42):
    random.seed(seed)
    n = img.shape[0]
    
    angle = random.choice([0, 90, 180, 270])
    
    rot_img = rotate_image(img, [angle])[0]
    rot_gt = rotate_image(gt, [angle])[0]
    
    if random.randrange(2) == 1:
        rot_img = flip_image(rot_img)
        rot_gt = flip_image(rot_gt)
                           
    while True:
        x = random.randrange(n - 80)
        y = random.randrange(n - 80)
        yield (
            rot_img[x:x+80,y:y+80,:],
            rot_gt[x+2*patch_size:x+3*patch_size,
                   y+2*patch_size:y+3*patch_size]
        )


    
# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[i:i+w, j:j+h] = labels[idx]
            idx = idx + 1
    return im
            
            
def img_to_submission_strings(labels, img_number, w, h):
    """outputs the strings that should go into the submission file for a given image"""
    label_img = labels.reshape(w//patch_size,h//patch_size)
    for j in range(0, w, patch_size):
        for i in range(0, h, patch_size):
            label = label_img[i//patch_size][j//patch_size]
            if label > foreground_threshold:
                sub_lab = 1
            else:
                sub_lab = 0
                
            yield("{:03d}_{}_{},{}".format(img_number, j, i, sub_lab))

            
def image_to_inputs(img, patch_size):
    rows, cols, _ = img.shape
    
    patches = [
        
            img[i-2*patch_size:i+3*patch_size, j-2*patch_size:j+3*patch_size, :]
        
        for i in range(2*patch_size, rows - 3*patch_size+1, patch_size)
        for j in range(2*patch_size, cols - 3*patch_size+1, patch_size)
    ]

    return patches


def disp_img_pred(img, pred):
    w = img.shape[0]
    h = img.shape[1]
    predicted_im = label_to_img(w, h, patch_size, patch_size, pred)
    new_img = make_img_overlay(img, predicted_im)
    fig1 = plt.figure(figsize=(10,10))
    plt.imshow(new_img)
    plt.show()


def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255
    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img
