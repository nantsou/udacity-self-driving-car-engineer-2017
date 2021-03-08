# -*- coding: utf-8 -*-
# import required modules
from skimage.feature import hog
import cv2
import numpy as np
from skimage.io import imread
import VehicleDetector.utils as utils

"""
cv2.anything() --> use (width, height)
image.anything() --> use (height, width)
numpy.anything() --> use (height, width)
"""

class FeatureExtractor(object):
    def __init__(self, orient=9, ppc=8, cpb=2, cspace='RGB', ssize=(32, 32)):
        """
        ppc: pixels per cell
        cpb: cells per block
        cspace: color space
        ssize: size for spatial color binning
        """
        # initialize the parameters
        self.orient = None
        self.ppc = None
        self.cpb = None
        self.cspace = None
        self.ssize = None

        # set the parameters if the initial values are given.
        self.set_params(orient, ppc, cpb, cspace, ssize)

    def set_orient(self, orient):
        self.orient = orient

    def set_px_per_cell(self, ppc):
        self.ppc = ppc

    def set_cells_per_block(self, cpb):
        self.cpb = cpb
    
    def set_color_space(self, cspace):
        self.cspace = cspace
    
    def set_spatial_binning_size(self, ssize):
        self.ssize = ssize

    def set_params(self, orient=None, ppc=None, cpb=None, cspace=None, ssize=None):
        if orient is not None:
            self.set_orient(orient)
        
        if ppc is not None:
            self.set_px_per_cell(ppc)
        
        if cpb is not None:
            self.set_cells_per_block(cpb)
        
        if cspace is not None:
            self.set_color_space(cspace)
        
        if ssize is not None:
            self.set_spatial_binning_size(ssize)
    
    def __is_image_path(self, image):
        return isinstance(image, str)

    def get_hog(self, ch, orient=9, ppc=8, cpb=2, viz=False, fv=True):
        """
        viz:    determine if return visualized hog feature
        fv:     determine if return hog feature vector
        """
        if viz == True:
            feature, image = hog(ch, orientations=orient, pixels_per_cell=(ppc, ppc),
                                 cells_per_block=(cpb, cpb), visualise=True, feature_vector=fv)
            return feature, image
        else:
            feature = hog(ch, orientations=orient, pixels_per_cell=(ppc, ppc),
                                 cells_per_block=(cpb, cpb), visualise=False, feature_vector=fv) 
            return feature

    def get_color_hist(self, image, bins=32, bins_range=(0, 256)):
        n_ch = image.shape[2] if len(image.shape) == 3 else 1
        if n_ch > 1:
            return np.concatenate([np.histogram(image[:,:,i], bins=bins, range=bins_range)[0] for i in range(n_ch)], axis=0)
        else:
            return np.histogram(image[:,:], bins=bins, range=bins_range)[0]
    
    def get_spatial_binning(self, image, size=(32, 32)):
        return cv2.resize(image, size).ravel()

    def extract_single_feature(self, image):
        # copy the image for getting hog features
        hog_image = np.copy(image)
        hog_image = utils.convert_color_space(hog_image, cspace=self.cspace)
        features = []
        chs = cv2.split(hog_image)
        for ch in chs:
            features.append(self.get_hog(ch, orient=self.orient, ppc=self.ppc, cpb=self.cpb))
        # get color histogram features
        features.append(self.get_color_hist(image))
        # get spatial binning features
        features.append(self.get_spatial_binning(image))
        # concatenate all the features as the output
        return np.concatenate(features, axis=0)
    
    def extract_features(self, images):
        # create a list to append the feature
        features = []
        # iterate through the images list
        for image in images:
            if self.__is_image_path(image):
                feature_image = imread(image)
            else:
                feature_image = image
            feature = self.extract_single_feature(feature_image)
            features.append(feature)
        
        return features
