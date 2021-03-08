# -*- coding: utf-8 -*- 
from os import path
import argparse
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

def regenerate_sample_log(log, origin, steering_adjust):

    if origin:
        return log.loc[:, ['center', 'steering']].rename(columns={'center': 'file_name'})

    cameras = ['center', 'left', 'right']
    tmps = None

    for camera in cameras:
        tmp = log.loc[:, [camera, 'steering']].rename(columns={camera: 'file_name'})
        
        # generate the data by slightly perturbing the sterring angles whoes absoulte values are greater than 0.15.
        ## get the data whoes steering angles are greater than 0.15.
        tmp_perturb = tmp.loc[abs(tmp['steering']) > 0.01].reset_index(drop=True)
        ## create the perturbing factor with the size same as temp_perturb
        factor = pd.DataFrame((1.0 + np.random.uniform(-1, 1, size=(tmp_perturb.shape[0]))/30.), columns=['value'])
        ## apply perturbing factor on the targeted data
        tmp_perturb.loc[:, 'steering'] *= factor['value']
        ## append perturbed data to tmp
        tmp = tmp.append(tmp_perturb, ignore_index=True)

        tmp_perturb = None


        # generate the data by flipping the images whoes steering angles are greater 0.10
        ## add flip flag to the data frame
        tmp.loc[:, 'flip']=False
        ## get the data whoes steering angles are greater than 0.10.
        tmp_flip = tmp.loc[abs(tmp['steering']) > 0.01].reset_index(drop=True)
        ## set flipping flag to the targeted data
        tmp_flip.loc[:, 'flip']=True
        ## append flip data to tmp
        tmp = tmp.append(tmp_flip, ignore_index=True)
        tmp_flip = None

        # reset the steering angle to siumlate the recovery if it is left camera or right camera
        if camera == 'left':
            tmp.loc[:, 'steering'] += steering_adjust
        elif camera == 'right':
            tmp.loc[:, 'steering'] -= steering_adjust

        ## concat the data frames
        tmps = pd.concat([tmps, tmp]).reset_index(drop=True)
        tmp = None

    return tmps

def get_feeding_data(row):
    angle = row['steering']
    image = plt.imread(path.join(args.data_path, row['file_name'].strip()))
    #if row['flip']:
    #    angle = -angle

    return image, angle

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate checking data')
    parser.add_argument('--data_path', type=str, default="data", help='the path where the original data is')
    parser.add_argument('--is_origin', action='store_true', default=False)
    parser.add_argument('--steering_adjust', type=float, default=0.25)
    parser.set_defaults(is_origin=True)
    args = parser.parse_args()
    
    base_path = args.data_path

    log = pd.read_csv(path.join(base_path, 'driving_log.csv'), usecols=[0, 1, 2, 3])
    log = regenerate_sample_log(log, args.is_origin, args.steering_adjust)
    log = log.sample(frac=1, random_state=1).reset_index(drop=True)


    x = np.empty([20, 160, 320, 3])
    y = np.empty([20])

    i = 0

    for index, row in log.iterrows():
        if i == 20:
            break
        elif i == 5:
            row['file_name'] = row['file_name'].replace('center', 'left')
        elif i == 16:
            row['file_name'] = row['file_name'].replace('center', 'right')
        x[index], y[index] = get_feeding_data(row)
        i = i+1

    np.save("x.data.origin", x)
    np.save("y.data.origin", y)
