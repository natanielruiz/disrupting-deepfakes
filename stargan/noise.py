import numpy as np
import os
import cv2
from PIL import Image

def PIL_to_cv2(image):
    image = np.array(image)
    image = image[:,:,::-1].copy()
    return image

def cv2_to_PIL(image):
    image = image[:,:,::-1].copy().astype(np.uint8)
    image = Image.fromarray(image)
    return image

def noisy(noise_typ, image):
    image = PIL_to_cv2(image)
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.2
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return cv2_to_PIL(noisy)
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return cv2_to_PIL(out)
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return cv2_to_PIL(noisy)
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return cv2_to_PIL(noisy)