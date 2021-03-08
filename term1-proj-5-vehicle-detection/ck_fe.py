# -*- coding: utf-8 -*-
# import required modules
import os
import cv2
import numpy as np
import pickle
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import random
# import VehicleDector module
from VehicleDector import utils
from VehicleDector.featureextractor import FeatureExtractor

def save_single_image(image, fname, row=1, col=1):
    fig = plt.figure()
    gs = gridspec.GridSpec(row, col)
    gs.update(wspace=0., hspace=0.)
    plt.savefig(OUTPUT_IMG_PATH + '/viz/viz_{}.png'.format(fname), 
                bbox_inches='tight',
                transparent="True", 
                pad_inches=0)

def set_ax(ax, image, cmap=None):
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(image, cmap=cmap)
    ax.set_aspect('auto')
    ax.title.set_visible(False)
    ax.axis('off')

    return ax

if __name__ == '__main__':

    car_images = glob('vehicles/*/*.png')
    non_car_images = glob('non-vehicles/*/*.png')

    idx = np.random.randint(0, len(car_images),size=1)
    car = cv2.imread(car_images[idx])
    car = cv2.cvtColor(car, cv2.COLOR_BGR2RGB)
    non_car = cv2.imread(non_car_images[idx])
    non_car = cv2.cvtColor(non_car, cv2.COLOR_BGR2RGB)

    # feature extract
    fe = FeatureExtractor()
    yuv_car = utils.convert_color_space(car, cspace='YUV')
    car_chs = cv2.split(yuv_car)

    car_features = []

    fig = plt.figure(figsize=(6, 2))
    gs = gridspec.GridSpec(1, 4)
    gs.update(wspace=0.1, hspace=0.0)
    #ax1 = fig.add_subplot(row, col, num)
    ax = plt.subplot(gs[0])
    ax = set_ax(ax, car)

    for i, ch in enumerate(car_chs):
        f, img = fe.get_hog(ch, viz=True, fv=True)
        car_features.append(f)
        ax = plt.subplot(gs[i+1])
        ax = set_ax(ax, img, cmap='gray')
    
    
    plt.savefig('output_images/test_fe.png', 
                bbox_inches='tight',
                transparent="True", 
                pad_inches=0)

    plt.show()

    yuv_non_car = utils.convert_color_space(non_car, cspace='YUV')
    non_car_chs = cv2.split(yuv_non_car)

    fig = plt.figure(figsize=(6, 2))
    gs = gridspec.GridSpec(1, 4)
    gs.update(wspace=0.1, hspace=0.0)
    #ax1 = fig.add_subplot(row, col, num)
    ax = plt.subplot(gs[0])
    ax = set_ax(ax, non_car)

    for i, ch in enumerate(non_car_chs):
        f, img = fe.get_hog(ch, viz=True, fv=True)
        car_features.append(f)
        ax = plt.subplot(gs[i+1])
        ax = set_ax(ax, img, cmap='gray')
    
    
    plt.savefig('output_images/test_fe.png', 
                bbox_inches='tight',
                transparent="True", 
                pad_inches=0)

    plt.show()