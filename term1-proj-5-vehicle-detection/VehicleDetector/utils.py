# -*- coding: utf-8 -*-
# import required modules
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

def convert_color_space(image, cspace='RGB'):
    if cspace == 'RGB':
        image = np.copy(image)
    elif cspace == 'HSV':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif cspace == 'HSV':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif cspace == 'LUV':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    elif cspace == 'HLS':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    elif cspace == 'YUV':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    elif cspace == 'YCrCb':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    elif cspace == 'LAB':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    elif cspace == 'GRAY':
        image = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), axis=-1)
    
    return image

def draw_bbox(image, bbox_list, color=(0, 0, 255), thick=6):
    # create a copy image
    draw_image = np.copy(image)
    # draw the boundary boxes of each target by using cv2.rectangle
    for bbox in bbox_list:
        cv2.rectangle(draw_image, bbox[0], bbox[1], color, thick)
    
    return draw_image

def draw_bbox_with_prob(image, win_out, win_prob, color=(0, 255, 0), thick=6):
    out_imge = np.copy(image)
    for i, bbox in enumerate(win_out):
        prob = win_prob[i][1]
        if prob < .6:
            color = (255,200,200)
        elif prob < .7:
            color = (255,100,100)
        elif prob < .8:
            color = (150,0,0)
        elif prob < .9:
            color = (100,50,0)
        elif prob <= 1:
            color = (50,0,0)
        # draw the box on the image
        cv2.rectangle(out_imge, bbox[0], bbox[1], color, thick)
    
    return out_imge

def get_scaler(car_features, non_car_features):
    # Create an array stack of feature vectors
    X = np.vstack((car_features, non_car_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)

    return X_scaler, X

def set_win_set_param(x_start_stop, y_start_stop, xy_window, xy_overlap):
    param = {}
    param['x_start_stop'] = x_start_stop
    param['y_start_stop'] = y_start_stop
    param['xy_window'] = xy_window
    param['xy_overlap'] = xy_overlap
    return param

def slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # if x_start_stop / y_start_stop is not defined, set them to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = image.shape[1]
    # define y_start_stop
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = image.shape[0]
    # compuate the span of region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # compute the pixels per step in x and y
    nx_px_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_px_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))

    # compute the number of windows in x and y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_px_per_step)
    ny_windows = np.int((yspan-nx_buffer)/ny_px_per_step)

    # initial a list to append the windows
    win_list = []
    # Loop through finding x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate each window position
            startx = xs*nx_px_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_px_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            win_list.append(((startx, starty), (endx, endy)))
    return win_list

def dump_content(fname, content):
    with open(fname, 'wb') as f:
        joblib.dump(content, f)

def dump_clf_and_feature_list(clf, feature, labels):
    dump_content('my_clf.p', clf)
    dump_content('my_feature.p', feature)
    dump_content('my_lables.p', labels)

def load_content(fname):
    with open(fname, 'rb') as f:
        tmp = joblib.load(f)
    
    return tmp
