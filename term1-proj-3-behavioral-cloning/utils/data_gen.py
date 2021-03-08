# -*- coding: utf-8 -*- 
# import required libraries
from os import path
import argparse
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# import my utils for data augmentation
from preprocessor import (crop_image, 
                           resize_image, 
                           grayscale_hsv, 
                           grayscale_yuv, 
                           grayscale_cv2)

def augment_log(log, args):

    cameras = ['center', 'left', 'right']
    tmps = None

    for camera in cameras:
        ## extract file name of camera and steering angle.
        tmp = log.loc[:, [camera, 'steering']]
        ## replace 'direction' of camera to 'filename'
        tmp = tmp.rename(columns={camera: 'file_name'})

        ## generate the data by slightly perturbing the sterring angles.
        ### the data whoes steering angles are greater than "perturb" is used
        ### to generate additional data.
        ### 0.01 is used for both model and tiny model.
        if args.perturb >= 0.0:
            tmp_perturb = tmp.loc[abs(tmp['steering']) > args.perturb].reset_index(drop=True)
            ### create the perturbing factor with the size same as temp_perturb
            factor = pd.DataFrame(
                (1.0 + np.random.uniform(-1, 1, size=(tmp_perturb.shape[0]))/30.), 
                columns=['value'])
            ### apply perturbing factor on the targeted data
            tmp_perturb.loc[:, 'steering'] *= factor['value']
            ### append perturbed data to tmp
            tmp = tmp.append(tmp_perturb, ignore_index=True)
            tmp_perturb = None

        ## generate the data by flipping the images 
        ### the data whoes steering angles are greater than "perturb" is used
        ### to generate additional data.
        ### 0.01 is used for both model and tiny model.
        if args.flip >= 0.0:
            ### add flip flag to the data frame
            tmp.loc[:, 'flip']=False
            ### the data whoes steering angles are greater than 0.01 is used in this project.
            tmp_flip = tmp.loc[abs(tmp['steering']) > args.flip].reset_index(drop=True)
            ### set flipping flag to the targeted data
            tmp_flip.loc[:, 'flip']=True
            ### append flip data to tmp
            tmp = tmp.append(tmp_flip, ignore_index=True)
            tmp_flip = None

        ## siumlate the recovery by adding extra angle to steering angle of
        ## left and right cameras.
        ### 0.25 is used for both model and tiny model.
        if args.recovery_angle:
            if camera == 'left':
                tmp.loc[:, 'steering'] += args.recovery_angle
            elif camera == 'right':
                tmp.loc[:, 'steering'] -= args.recovery_angle

        ### concat the data frames
        tmps = pd.concat([tmps, tmp]).reset_index(drop=True)
        tmp = None

    return tmps

def preprocess_image(image, args):

    ## crop image
    ### (crop_from, crop_to): (53, 133) for model. 
    ### no cropping for tiny model.
    ### crop when the range of cropping is not 160 which original size.
    if args.crop_from != 0 or args.crop_to != 160:
        image = crop_image(image, args.crop_from, args.crop_to)

    ## resize image
    ### (new_h, new_w): (16, 32) for tiny model 
    ### and (16, 64) for model.
    ### resize when new_h and new_w are not the original sizes.abs
    if args.new_h != 160 or args.new_w != 320:
        image = resize_image(image, args.new_h, args.new_w)

    ## grayscale images
    ### grayscale_hsv is used for both model and tiny model
    ### grayscale when grayscaling method is given.
    if args.gray_method == 'hsv':
        image = grayscale_hsv(image)
    elif args.gray_method == 'yuv':
        image = grayscale_yuv(image)
    elif args.gray_method == 'cv2':
        image = grayscale_cv2(image)
    
    ### set n_channel to 1 if grayscaling method is given
    ### else set n_channel to 3 which is the original number of the channels.
    n_channel = 1 if args.gray_method else 3

    ## reshape the image array for inputting the cnn model
    return image.reshape(args.new_h, args.new_w, n_channel)

def get_augmented_data(row, args):
    angle = row['steering']
    ## load image data with regenerate logs
    image = plt.imread(path.join(args.data_path, row['file_name'].strip()))

    if row['flip']:
        angle = -angle
        image = cv2.flip(image, 1)
    
    ## preprocess the image data
    image = preprocess_image(image, args)

    return image, angle

# validation for argsparse
def check_positive_float(value):
    val = float(value)
    if val < 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive float value" % value)
    return val

def check_grayscaling_method(value):
    val = str(value)
    if value not in ['hsv', 'yuv', 'cv2', '']:
        raise argparse.ArgumentTypeError("%s is an invalid method for grayscaling" % value)
    return val

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate checking data')
    parser.add_argument(
        '--data_path', type=str,default="data", 
        help='the path where the image data and log file are.')
    parser.add_argument(
        '--suffix', type=str, default="train", 
        help='the suffix added to output file name.')
    parser.add_argument(
        '--crop_from', type=int, default=0, 
        help="upper value for cropping the image. from 0 to 160")
    parser.add_argument(
        '--crop_to', type=int, default=160, 
        help="lower value for cropping the image. from 0 to 160")
    parser.add_argument(
        '--new_h', type=int, default=160, 
        help="height for resizing the image.")
    parser.add_argument(
        '--new_w', type=int, default=320, 
        help="width for resizing the image.")
    parser.add_argument(
        '--gray_method', type=check_grayscaling_method, default="", 
        help="the method for grayscaling the image. option: hsv, yuv, cv2")
    parser.add_argument(
        '--recovery_angle', type=check_positive_float, default=0.00, 
        help="the angle for simulating the recovery.")
    parser.add_argument(
        '--perturb', type=check_positive_float, default=0.00, 
        help="the lowest value to extract the data for perturbing angles.")
    parser.add_argument(
        '--flip', type=check_positive_float, default=0.00, 
        help="the lowest value to extract the data for flipping images.")
    args = parser.parse_args()
    
    ## load driving logs
    log = pd.read_csv(path.join(args.data_path, 'driving_log.csv'), 
                      usecols=[0, 1, 2, 3])
    ## augment log information.
    log = augment_log(log, args)
    ## shuffle the order of the data.
    log = log.sample(frac=1, random_state=1).reset_index(drop=True)

    n_channel = 1 if args.gray_method else 3

    ## initialate the container for the training data
    x = np.empty([log.shape[0], args.new_h, args.new_w, n_channel])
    y = np.empty([log.shape[0]])
    for index, row in log.iterrows():
        x[index], y[index] = get_augmented_data(row, args)
    
    np.save("x.data.{}".format(args.suffix), x)
    np.save("y.data.{}".format(args.suffix), y)
