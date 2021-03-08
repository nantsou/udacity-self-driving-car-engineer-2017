# -*- coding: utf-8 -*-
# import required modules
import cv2
import numpy as np

# import my modules
from VehicleDetector.featureextractor import FeatureExtractor
from VehicleDetector.heatmap import HeatMap
import VehicleDetector.utils as utils

class VehicleDetector(object):
    def __init__(self, clf=None, x_scaler=None, fe=None, heatmap_threshold=1, n_frame=2, win_set_params=None):
        """
        clf:        classifier
        x_scaler:   X_scaler used to normalize the features
        fe:         feature extractor
        hm:         heatmap
        """
        self.frame_cnt = 0
        self.vehicle_cnt = 0
        self.skip_cnt = 3
        self.clf = clf
        self.x_scaler = x_scaler
        self.fe = fe
        self.hm = HeatMap(n_frame=n_frame, threshold=heatmap_threshold)
        self.win_set_params = win_set_params
        self.full_scan_win_set = None
        self.bbox_list = None

    def set_classifier(self, clf):
        self.clf = clf

    def set_x_scaler(self, x_scaler):
        self.x_scaler = x_scaler

    def set_feature_extractor(self, fe):
        self.fe = fe

    def set_heatmap(self,hm):
        self.hm = hm

    def get_win_set(self, image, win_set_params=None):
        win_set = []
        if win_set_params is None or len(win_set_params) == 0:
            win_set = utils.slide_window(image, x_start_stop=[0, image.shape[1]], y_start_stop=[340, 680], xy_window=(96, 96), xy_overlap=(0.8, 0.8))
        else:
            for param in win_set_params:
                win_set += utils.slide_window(image,
                                              x_start_stop=param['x_start_stop'], 
                                              y_start_stop=param['y_start_stop'], 
                                              xy_window=param['xy_window'], 
                                              xy_overlap=param['xy_overlap'])
        return win_set

    def get_small_region(self, bbox_list):
        x_start = 1280
        x_stop = 0
        y_start = 720
        y_stop = 0

        if len(bbox_list) > 0:
            for bbox in bbox_list:
                x1 = bbox[0][0]
                y1 = bbox[0][1]
                x2 = bbox[1][0]
                y2 = bbox[1][1]

                x_start = np.minimum(x_start, np.minimum(x1, x2))
                x_stop = np.maximum(x_stop, np.maximum(x1, x2))
                y_start = np.minimum(y_start, np.minimum(y1, y2))
                y_stop = np.maximum(y_stop, np.maximum(y1, y2))

            return x_start, x_stop, y_start, y_stop

    def get_small_region_win_set(self, image, x_start, x_stop, y_start, y_stop):
        # define the margins of x and y for detecting the area around the detected vehicles
        x_margin = 25
        y_margin = 10

        # prevent x_start which outside the image
        x_start = np.maximum(x_start - x_margin, 0)
        x_stop = np.maximum(x_stop + x_margin, image.shape[1])
        # prevent y_start which is higher than half height of the image
        y_start = np.maximum(y_start - y_margin, image.shape[0]//2)
        # prevent y_stop which is lower than the hood of car
        y_stop = np.minimum(y_stop + y_margin, np.int(image.shape[0] * 0.92))

        tracking_win_set = []
        for size in [64, 80, 100, 120, 150]:
            tracking_win_set += utils.slide_window(image,
                                              x_start_stop=[x_start, x_stop],
                                              y_start_stop=[y_start, y_stop],
                                              xy_window=(size, size),
                                              xy_overlap=(0.8, 0.8))
        return tracking_win_set

    def get_win_images(self, image, win_set):
        win_images = []
        # loop through the windows
        for win in win_set:
            # get the image of the windows
            win_image = image[win[0][1]:win[1][1], win[0][0]:win[1][0], :]
            # resize the window to match the images used in model training.
            # the default size of the images used in model training is (64, 64)
            win_image = cv2.resize(win_image, (64,64))
            win_images.append(win_image)
        return win_images

    def predict(self, win_features):
        X_feature = np.array(win_features).astype(np.float64).reshape(1, -1)
        X_feature_scaled = self.x_scaler.transform(X_feature)
        predictions = self.clf.predict(X_feature_scaled)
        probabilities = self.clf.predict_proba(X_feature_scaled)
        return predictions, probabilities

    def detect(self, win_set, win_images):
        win_out = []
        prob_out = []
        for idx, win_image in enumerate(win_images):
            feature = self.fe.extract_single_feature(win_image)
            pred, prob = self.predict(feature)

            if pred == 1:
                win_out.append(win_set[idx])
                prob_out.extend(prob)
        return win_out, prob_out

    def label_to_bbox(self, labels):
        bbox_list = []
        # iterate through all detected cars
        for vehicle_cnt in range(1, labels[1] + 1):
            # find pixels with each car_number label value
            nonzero = (labels[0] == vehicle_cnt).nonzero()
            # identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # store bbox
            bbox_list.append(bbox)
        return bbox_list

    def process(self, image, bbox_only=False):
        # count the frame
        self.frame_cnt += 1
        # copy input image used in final draw method
        input_image = np.copy(image)
        # set variable to control whether skip the detection or not
        skip_detect = False
        # initialize the win_set for full scan
        if self.full_scan_win_set is None:
            self.full_scan_win_set = self.get_win_set(input_image, self.win_set_params)
        # set local window set to full scan window set as default
        win_set = self.full_scan_win_set
        # set win_set to tracking window set if it is either 1st frame nor every 5th frame
        if self.frame_cnt != 1 and np.remainder(self.frame_cnt, 5) > 0:
            # if there are detected vehicles in previous frame then update the tracking window set
            if self.vehicle_cnt > 0:
                x_start, x_stop, y_start, y_stop = self.get_small_region(self.bbox_list)
                win_set = self.get_small_region_win_set(image, x_start, x_stop, y_start, y_stop)
            else:
                # skip the scan if no vehicles detected in previous frames for 3 times
                if self.skip_cnt > 0:
                    self.skip_cnt -= 1
                    skip_detect = True
                # reset the skip_cnt with 3 and carry out full scan
                else:
                    self.skip_cnt = 3
        
        if skip_detect == False:
            # get the images of each window
            win_images = self.get_win_images(image, win_set)
            # get the predecting results
            win_out, win_prob = self.detect(win_set, win_images)
            # get the heatmap labels
            labels = self.hm.get_heatmap_labels(image, win_out)
            # get current number of the detected vehicles
            self.vehicle_cnt = labels[1]
            # get the boxes of detected vehicles
            self.bbox_list = self.label_to_bbox(labels)

            if bbox_only:
                return self.bbox_list
            else:
                return utils.draw_bbox(input_image, self.bbox_list, color=(0, 255, 0))
        else:
            if bbox_only:
                return []
            else:
                return input_image