# -*- coding: utf-8 -*-

from os import path
import numpy as np
import cv2

from preprocessor import (crop_image, 
                          resize_image, 
                          grayscale_hsv, 
                          grayscale_cv2, 
                          grayscale_yuv)

def preprocess(image, grayscale, is_tiny=False):
    
    if is_tiny:
        new_h = 16
        new_w = 32
    else:
        new_h = 16
        new_w = 64
    
    if not is_tiny:
        ## crop image
        image = crop_image(image, 53, 133)

    ## resize image
    image = resize_image(image, new_h, new_w)

    ## convert RGB image to grey image and normalize the result
    image = grayscale(image)

    if not is_tiny:
        image.reshape(new_h, new_w)

    return image


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12,3))

    img = plt.imread('../assets/images/center.jpg')
    img_tmp = crop_image(img, 53, 133)
    fig.add_subplot(1,4,1)
    plt.axis('off')
    plt.title("origin", fontsize=12)
    plt.imshow(img_tmp)


    prep_img = preprocess(img, grayscale_cv2)
    fig.add_subplot(1,4,2)
    plt.axis('off')
    plt.title("grayscale_cv2", fontsize=12)
    plt.imshow(prep_img, cmap='gray')

    prep_img = preprocess(img, grayscale_hsv)
    fig.add_subplot(1,4,3)
    plt.axis('off')
    plt.title("grayscale_hsv", fontsize=12)
    plt.imshow(prep_img, cmap='gray')

    prep_img = preprocess(img, grayscale_yuv)
    fig.add_subplot(1,4,4)
    plt.axis('off')
    plt.title("grayscale_yuv", fontsize=12)
    plt.imshow(prep_img, cmap='gray')
    
    plt.tight_layout()
    plt.savefig('../assets/images/grayscaling.png')
    plt.show()
