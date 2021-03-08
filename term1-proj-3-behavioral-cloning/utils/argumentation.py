# -*- coding: utf-8 -*-

from os import path
import numpy as np
import cv2


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2

    fig = plt.figure(figsize=(12,6))

    x = np.load('x.data.origin.npy')
    y = np.load('y.data.origin.npy')
    x = x.astype('uint8')

    # origin
    img = x[8]
    angle = y[8]
    fig.add_subplot(3,3,1)
    plt.axis('off')
    plt.title("origin: {}".format(str(np.round(angle,5))), fontsize=12)
    plt.imshow(img)

    # perturb
    fig.add_subplot(3,3,2)
    plt.axis('off')
    factor = (1.0 + np.random.uniform(-1, 1)/30.)
    p_angle = angle * factor
    plt.title("perturb: {}".format(str(np.round(p_angle,5))), fontsize=12)
    plt.imshow(img)

    # flip
    f_angle = -angle
    flip_image = cv2.flip(img, 1)
    fig.add_subplot(3,3,3)
    plt.axis('off')
    plt.title("flip: {}".format(str(np.round(f_angle,5))), fontsize=12)
    plt.imshow(flip_image)

    # recovery left
    img = x[5]
    angle = y[5]
    fig.add_subplot(3,3,4)
    plt.axis('off')
    plt.title("origin-left: {}".format(str(np.round(angle,2))), fontsize=12)
    plt.imshow(img)

    fig.add_subplot(3,3,5)
    plt.axis('off')
    plt.title("recovery-left: {}".format(str(np.round(angle + 0.25 ,2))), fontsize=12)
    plt.imshow(img)

    # recovery right
    img = x[16]
    angle = y[16]
    fig.add_subplot(3,3,7)
    plt.axis('off')
    plt.title("origin-right: {}".format(str(np.round(angle,2))), fontsize=12)
    plt.imshow(img)

    fig.add_subplot(3,3,8)
    plt.axis('off')
    plt.title("origin-right: {}".format(str(np.round(angle - 0.25,2))), fontsize=12)
    plt.imshow(img)
    
    plt.tight_layout()
    plt.savefig('../assets/images/argumentation.png')
    plt.show()
    
