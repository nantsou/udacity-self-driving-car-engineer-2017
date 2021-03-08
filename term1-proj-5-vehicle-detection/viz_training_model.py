# -*- coding: utf-8 -*-
# import required modules
import os
import cv2
import numpy as np
import pickle
# import VehicleDector module
from VehicleDetector import utils
from VehicleDetector.featureextractor import FeatureExtractor

# define basic info
ROOT = os.path.dirname(os.path.realpath(__file__))
CAR_PATH_ROOT = os.path.join(ROOT, 'vehicles')
NON_CAR_PATH_ROOT = os.path.join(ROOT, 'non-vehicles')

x_scaler_path = 'outputs/x_scaler.p'
# load scaler
x_scaler = utils.load_content(x_scaler_path)

# define color channel name
ch_name_yuv = {
    0: 'Y',
    1: 'U',
    2: 'V'
}
ch_name_rgb = {
    0: 'R',
    1: 'G',
    2: 'B'
}
# define util methods
def save_single_image(image, fname, cmap=None):
    fig = plt.figure(figsize=(4.5, 4.5))
    plt.imshow(image, cmap=cmap)
    a=fig.gca()
    a.set_frame_on(False)
    a.set_xticks([]); a.set_yticks([])
    plt.axis('off')
    plt.savefig('output_images/viz_tm/tm_{}.png'.format(fname), 
                bbox_inches='tight',
                transparent="True", 
                pad_inches=0)

def save_single_bar_plot(bincen, hist_res, color, fname):
    fig = plt.figure(figsize=(6, 4.5))
    plt.bar(bincen, hist_res, color=color)
    plt.xlim(0, 256)
    a=fig.gca()
    a.set_frame_on(False)
    a.set_xticks([]); a.set_yticks([])
    plt.axis('off')
    plt.savefig('output_images/viz_tm/tm_{}.png'.format(fname), 
                bbox_inches='tight',
                transparent="True", 
                pad_inches=0)

def save_single_plot(feature, color, fname):
    fig = plt.figure(figsize=(6, 4.5))
    plt.plot(feature, color=color)
    a=fig.gca()
    a.set_frame_on(False)
    a.set_xticks([]); a.set_yticks([])
    plt.axis('off')
    plt.savefig('output_images/viz_tm/tm_{}.png'.format(fname), 
                bbox_inches='tight',
                transparent="True", 
                pad_inches=0)

def color_hist(image, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(image[:,:,0], bins=nbins, range=bins_range)
    ghist = np.histogram(image[:,:,1], bins=nbins, range=bins_range)
    bhist = np.histogram(image[:,:,2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return rhist, ghist, bhist, bin_centers, hist_features

if __name__ == '__main__':
    from glob import glob
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import random

    car_images = glob(CAR_PATH_ROOT + '/*/*.png')
    non_car_images = glob(NON_CAR_PATH_ROOT + '/*/*.png')

    idx = np.random.randint(0, len(car_images),size=1)
    # use the pictures whoes index is 8399 to visualize the process
    car = cv2.imread(car_images[idx])
    car = cv2.cvtColor(car, cv2.COLOR_BGR2RGB)

    non_car = cv2.imread(non_car_images[idx])
    non_car = cv2.cvtColor(non_car, cv2.COLOR_BGR2RGB)

    car = np.copy(car)
    save_single_image(car, 'input_car')

    non_car = np.copy(non_car)
    save_single_image(non_car, 'input_non_car')

    # feature extract
    fe = FeatureExtractor()
    yuv_car = utils.convert_color_space(car, cspace='YUV')
    car_chs = cv2.split(yuv_car)
    # hog features
    for i, ch in enumerate(car_chs):
        fname = 'hog_car_{}'.format(ch_name_yuv[i])
        _, img = fe.get_hog(ch, viz=True, fv=True)
        save_single_image(img, fname, cmap='gray')
    
    yuv_non_car = utils.convert_color_space(non_car, cspace='YUV')
    non_car_chs = cv2.split(yuv_non_car)

    for i, ch in enumerate(non_car_chs):
        fname = 'hog_non_car_{}'.format(ch_name_yuv[i])
        _, img = fe.get_hog(ch, viz=True, fv=True)
        save_single_image(img, i, cmap='gray')
    
    # color histogram features
    ## car
    rhist, ghist, bhist, bin_centers, hist_features = color_hist(car)
    for i, hist in enumerate([rhist, ghist, bhist]):
        fname = 'c_hist_car_{}'.format(ch_name_rgb[i])
        color=[[0.0, 0.0, 0.0, 1.0]]
        color[0][i] = 1.0
        save_single_bar_plot(bin_centers, hist[0], color, fname)
    
    ## non-car
    rhist, ghist, bhist, bin_centers, hist_features = color_hist(non_car)
    for i, hist in enumerate([rhist, ghist, bhist]):
        fname = 'c_hist_non_car_{}'.format(ch_name_rgb[i])
        color=[[0.0, 0.0, 0.0, 1.0]]
        color[0][i] = 1.0
        save_single_bar_plot(bin_centers, hist[0], color, fname)

    # spatial binning feature
    ## car
    car_features = cv2.resize(car, (32, 32)).ravel()
    fname='spatial_binning_car'
    save_single_plot(car_features, color='gray', fname=fname)
    ## non-car
    non_car_features = cv2.resize(non_car, (32, 32)).ravel()
    fname='spatial_binning_non_car'
    save_single_plot(non_car_features, color='gray', fname=fname)

    # overall raw features
    overall_raw_car_feature = fe.extract_single_feature(car)
    save_single_plot(overall_raw_car_feature, color='gray', fname='overall_raw_car_feature')
    overall_raw_non_car_feature = fe.extract_single_feature(non_car)
    save_single_plot(overall_raw_non_car_feature, color='gray', fname='overall_raw_non_car_feature')

    ## use the x_scaler created by model training
    # overall scaled features
    overall_raw_car_feature = np.array(overall_raw_car_feature).astype(np.float64).reshape(1, -1)
    overall_scaled_car_feature = x_scaler.transform(overall_raw_car_feature)
    save_single_plot(overall_scaled_car_feature[0], color='gray', fname='overall_scaled_car_feature')

    overall_raw_non_car_feature = np.array(overall_raw_non_car_feature).astype(np.float64).reshape(1, -1)
    overall_scaled_non_car_feature = x_scaler.transform(overall_raw_non_car_feature)
    save_single_plot(overall_scaled_non_car_feature[0], color='gray', fname='overall_scaled_non_car_feature')
    plt.show()