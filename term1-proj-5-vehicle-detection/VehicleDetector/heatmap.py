# -*- coding: utf-8 -*-
# import required modules
import cv2
import numpy as np
from scipy.ndimage.measurements import label

class HeatMap(object):
    def __init__(self, image=None, n_frame=1, threshold=1):
        self.n_frame = n_frame
        self.threshold = threshold
        self.avg_heat = None
        self.heats = []
            
    def weight_heat(self, cur_heat):
        if self.avg_heat is None:
            self.avg_heat = cur_heat
        else:
            self.avg_heat = self.avg_heat*2/3. + cur_heat*1/3.

    def add_heat(self, image, bbox_list):
        cur_heat = np.zeros_like(image[:,:,0]).astype(np.float)
        for box in bbox_list:
            cur_heat[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # store current heatmap
        self.heats.append(cur_heat)
        # remove oldest one if the number of stored heatmap is over n_frame
        if len(self.heats) > self.n_frame:
            self.heats.pop(0)

        # update average heatmap
        self.weight_heat(cur_heat)

    def apply_threshold(self, threshold):
        self.avg_heat[self.avg_heat <= threshold] = 0

    def label(self):
        return label(self.avg_heat)
    
    def get_heatmap_labels(self, image, bbox_list):
        self.add_heat(image, bbox_list)
        self.apply_threshold(self.threshold)
        return label(self.avg_heat)

