# -*- coding: utf-8 -*-
# import required modules
## basic modules
import os
import cv2
import numpy as np
import pickle
from glob import glob
import time
## classifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
## cross validation utils
from sklearn.model_selection import train_test_split

# import my modules
from VehicleDetector.featureextractor import FeatureExtractor
from VehicleDetector.utils import get_scaler, dump_content, load_content

# basic definition
OUT_PATH = 'outputs'

def need_extract_feature():
    have_car_feature = os.path.isfile(os.path.join(OUT_PATH, 'car_features.npy'))
    have_non_car_features = os.path.isfile(os.path.join(OUT_PATH, 'non_car_features.npy'))
    return not (have_car_feature and have_non_car_features)

if __name__ == "__main__":

    car_images = glob('vehicles/*/*.png')
    non_car_images = glob('non-vehicles/*/*.png')

    # get the features and labels of car and non-car
    ## if there are no features or labels files
    ## create FeatureExtractor object with the parameters to extract the features
    if need_extract_feature():
        fe = FeatureExtractor(orient=9, ppc=8, cpb=2, cspace="YUV", ssize=(32, 32))
        ## extract car features
        car_features = fe.extract_features(car_images)
        ## extract non-car features
        non_car_features = fe.extract_features(non_car_images)

        ## save features
        np.save(os.path.join(OUT_PATH, 'car_features.npy'), car_features)
        np.save(os.path.join(OUT_PATH, 'non_car_features.npy'), non_car_features)

        print('complated feature extracting')
    else:
        car_features = np.load(os.path.join(OUT_PATH, 'car_features.npy'))
        non_car_features = np.load(os.path.join(OUT_PATH, 'non_car_features.npy'))

    if not os.path.isfile(os.path.join(OUT_PATH, 'x_scaler.p')):
        # combine car_featrues and non-car features and labels them with car is 1 and non-car is 0
        ## build scaled features
        X_scaler, X = get_scaler(car_features, non_car_features)
        X_scaled = X_scaler.transform(X)
        ## save X_scaler
        dump_content(os.path.join(OUT_PATH, 'x_scaler.p'), X_scaler)
        ## bulid labels
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))
    else:
        X = np.vstack((car_features, non_car_features)).astype(np.float64)
        X_scaler = load_content(os.path.join(OUT_PATH, 'x_scaler.p'))
        X_scaled = X_scaler.transform(X)
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))
    
    print('completed getting features and labels')

    # prepare train dataset and test dataset for training model
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=7)

    # Use a linear SVC 
    svc = LinearSVC(C=1.0) 
    clf = CalibratedClassifierCV(svc) # to know the actual probability
    clf.fit(X_train, y_train)
    # Check the score of the SVC
    print('Train Accuracy of SVC = ', clf.score(X_train, y_train))
    print('Test Accuracy of SVC = ', clf.score(X_test, y_test))
    # Check the prediction time for a single sample
    prediction = clf.predict(X_test[0].reshape(1, -1))
    
    print()
    print('completed training model...')

    # save Linear SVC model
    dump_content(os.path.join(OUT_PATH, 'svc.p'), clf)
    print()
    print('completed saving model...')
