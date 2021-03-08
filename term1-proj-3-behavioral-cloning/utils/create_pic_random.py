# -*- coding: utf-8 -*-

from os import path
import numpy as np
import matplotlib.pyplot as plt

def randomly_show_results(x, y):
    fig = plt.figure(figsize=(16,6))
    for i in range(20):
        image = x[i]
        angle = y[i]
        
        plt.subplot(4, 5, i+1)
        plt.imshow(image, cmap='gray');
        plt.axis('off')
        plt.title(str(np.round(angle,2)))
    
    plt.tight_layout()
    plt.savefig('../assets/images/random_train_data.png')
    plt.show()

if __name__ == '__main__':

    x = np.load('x.data.train.npy')
    y = np.load('y.data.train.npy')
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2])

    randomly_show_results(x, y)