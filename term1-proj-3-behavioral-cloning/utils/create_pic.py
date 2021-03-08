# -*- coding: utf-8 -*-

from os import path
import numpy as np
import cv2

from preprocessor import crop_image, resize_image, grayscale_hsv

def preprocess(image, is_tiny=False):
    
    if is_tiny:
        new_h = 16
        new_w = 32
    else:
        # actually new_h shoud be 16, but I use black color for the parts which is cropped
        new_h = 32 
        new_w = 64
    
    if not is_tiny:
        cropped_image = np.zeros([image.shape[0], image.shape[1], 3], dtype='uint8')
        ## crop image
        cropped_image[53:133, :, :] = crop_image(image, 53, 133)
        image = cropped_image

    ## resize image
    image = resize_image(image, new_h, new_w)

    ## convert RGB image to grey image and normalize the result
    image = grayscale_hsv(image)

    if not is_tiny:
        image.reshape(new_h, new_w)

    return image


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8,4))

    img = plt.imread('../assets/images/center.jpg')
    fig.add_subplot(1,2,1)
    plt.axis('off')
    plt.title("origin", fontsize=12)
    plt.imshow(img)

    prep_img = preprocess(img, is_tiny=True)
    fig.add_subplot(1,2,2)
    plt.axis('off')
    plt.title("preprocessed", fontsize=12)
    plt.imshow(prep_img, cmap='gray')

    
    plt.tight_layout()
    plt.savefig('../assets/img/comparsion_tiny.png')
    plt.show()
