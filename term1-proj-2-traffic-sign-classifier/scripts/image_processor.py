import numpy as np
from sklearn.preprocessing import LabelBinarizer
import scipy.ndimage
import cv2
from sklearn.model_selection import train_test_split

# Preprocess image data

## Generate additional image data to make up the classes which have fewer image data.
## Generate additional image data by rotating the angle of images from -20 degree to 20 degree with the step 5 degree.
def generate_additional_image_data(features, labels):
    class_cnt = np.bincount(labels)
    max_class_n = np.max(class_cnt)

    ## Generate additional image data by rotating the angle of images from -20 degree to 20 degree with the step 5 degree.
    angles = [-20, -15, -10, -5, 5, 10, 15, 20]

    ## The classes whoes number of data is 3 times less than that of class with maximum number of data
    ## The adding data is up to the times as the same as the number of rotating angles.
    for i, cnt in enumerate(class_cnt):
        adding_times = min(int(max_class_n/cnt) - 1, len(angles))
        
        if adding_times <= 1:
            continue

        added_features = []
        added_labels = []
        target = np.where(labels == i)
        for j in range(adding_times):
            for feature in features[target]:
                added_features.append(scipy.ndimage.rotate(feature, angles[j], reshape=False))
                added_labels.append(i)

        features = np.append(features, added_features, axis=0)
        labels = np.append(labels, added_labels, axis=0)
    return features, labels

# Normalize training image data in between 0.1 to 0.9
def normalize(img):
    max_val = np.max(img)
    min_val = np.min(img)

    return 0.1 + np.divide((img - min_val)*0.8, (max_val - min_val))

# Change 3 channels(color) to 1 channel(grey)
## CLAHE (Contrast Limited Adaptive Histogram Equalization) which is suggested by Open CV 2
## As paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks" said,
## Y channel of yuv is enough. So, Y channel is processed only
def to_grey(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    Y, U, V = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6,6))
    Y = clahe.apply(Y)

    return normalize(Y)

# Preprocess image data with two methods above.
def to_grey_processor(features):
    # get number of dataset
    n = features.shape[0]
    size_w = features.shape[1]
    size_h = features.shape[2]

    # initialize 
    features_grey = np.zeros([n, size_w, size_h])

    for i in range(n):
        features_grey[i,] = to_grey(features[i])
    
    return features_grey

# Flatten Image dataset from 3 dimensions to 1 dimension
def flatten_dataset(features):
    n = features.shape[0]
    n_pix = features.shape[1]*features.shape[2]

    return np.reshape(features,[n,n_pix])
    

# Turn labels into numbers and apply One-Hot Encoding
def one_hot_encoding(labels):
    encoder = LabelBinarizer()
    encoder.fit(labels)
    return encoder.transform(labels)