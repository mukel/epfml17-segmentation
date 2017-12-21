# Helper functions
import os,sys
from PIL import Image
import re
import cv2
import imutils
import matplotlib.image as mpimg

import numpy as np


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

def reflect_border(im, patch_size, count) :
    bordersize = patch_size*count
    border=cv2.copyMakeBorder(im, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_REFLECT )
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
    n = min(limit, len(files)) # Load maximum 20 images
    print('Loading', n, 'training images')
    imgs = [load_image(image_dir + files[i]) for i in range(n)]
    gt_dir = root_dir + "groundtruth/"
    print('Loading', n, 'groundtruth images')
    gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]
    return (imgs, gt_imgs)

