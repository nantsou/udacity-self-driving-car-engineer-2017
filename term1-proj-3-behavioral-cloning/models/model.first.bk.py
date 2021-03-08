# -*- coding: utf-8 -*-

from os import path
import numpy as np
import cv2
import matplotlib.image as mpimg
from keras.optimizers import Adam
from keras.layers import Convolution2D as Conv2D, Input, Dropout, Flatten, Dense, MaxPooling2D, AveragePooling2D, ELU, Reshape
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
#from keras.preprocessing.image import img_to_array, load_img

#from utils.preprocessor import crop_image, resize_image, normalize, grayscale_hsv

# Basic parameters
NEW_H = 16
NEW_W = 32
N_CHANNEL = 1
BASE_DATA_PATH = 'data' # the folder is located at where the model.py (this file) is.


def regenerate_sample_log(log, steering_adjust=0.25):
    cameras = ['center', 'left', 'right']

    tmps = None

    for camera in cameras:
        tmp = log.loc[:, [camera, 'steering']].rename(columns={camera: 'file_name'})
        if camera == 'left':
            tmp.loc[:, 'steering'] += steering_adjust
        elif camera == 'right':
            tmp.loc[:, 'steering'] -= steering_adjust

        tmp.loc[:, 'flip']=False
        tmp_flip = tmp.loc[abs(tmp['steering']) >= 0.01]
        tmp_flip.loc[:, 'flip']=True
        
        tmps = pd.concat([tmps, tmp, tmp_flip])
        tmp = tmp_flip = None
    return tmps


    """ 
    curr_n = samples.shape[0]
    # Get samples whoes steering over 0.01
    sample_1 = samples[abs(samples['steering']) >= 0.01]
    # Get samples whoes steering over 0.05
    sample_2 = samples[abs(samples['steering']) >= 0.05]
    # Get samples whoes steering over 0.10
    sample_3 = samples[abs(samples['steering']) >= 0.10]
    # Get samples whoes steering over 0.15
    sample_4 = samples[abs(samples['steering']) >= 0.15]
    # Get samples whoes steering over 0.20
    sample_5 = samples[abs(samples['steering']) >= 0.20]


    while curr_n <= max_n_samples:
        # append samples above 
        samples = samples.append([sample_1, sample_2, sample_3, sample_4, sample_5])
        curr_n = samples.shape[0]
    """
    return samples

def preprocess_image(image):
    ## crop image
    ##image = crop_image(image, 53, 133)

    ## resize image
    image = resize_image(image, NEW_H, NEW_W)

    ## convert RGB image to grey image and normalize the result
    image = to_grey(image)

    return image

def augment_data(row):
    angle = row['steering']

    ## randomly choose the direction of the camera from center, left and right
    direction = np.random.choice(['center', 'left', 'right'])

    ## adjust steering angle of left or right camera to simulate recovery
    ## As the conversations in slack, 0.25 may be a good adjustment.
    if direction == 'left':
        angle += 0.25
    elif direction == 'right':
        angle -= 0.25
    
    ## load image data
    image = mpimg.imread(path.join(BASE_DATA_PATH, row[direction].strip()))
    #image = load_img(path.join(BASE_DATA_PATH, row[direction].strip()))
    #image = img_to_array(image)

    ## horizontally flip image
    if np.random.choice([True, False]):
        angle *= -1
        image = cv2.flip(image, 1)
    
    ## preprocess image with crop, resize, to grey and normalize
    image = preprocess_image(image)

    return image, angle

def get_feeding_data(row):
    angle = row['steering']
    ## load image data with regenerate logs
    image = mpimg.imread(path.join(BASE_DATA_PATH, row['file_name'].strip()))
    #image = load_img(path.join(BASE_DATA_PATH, row['file_name'].strip()))
    #image = img_to_array(image)

    if row['flip']:
        angle = -angle
        image = cv2.flip(image, 1)

    image = preprocess_image(image)

    return image, angle


def get_generator(data, batch_size=128):
    #total_n = data.shape[0]
    iter_n = 0

    while 1:
        start = iter_n*batch_size
        end = start + batch_size - 1

        batch_x = np.zeros((batch_size, NEW_H, NEW_W))
        #batch_x = np.zeros((batch_size, 40, 160, 3))
        batch_y = np.zeros(batch_size)
        batch_i = 0
        for _, row in data.loc[start:end].iterrows():
            batch_x[batch_i], batch_y[batch_i] = get_feeding_data(row)
            batch_i += 1
        
        #iter_n += 1
        ## Reset the iteration and shuffle to keep the training.
        #if iter_n * batch_size > total_n - batch_size:
        #    iter_n = 0
        #    data = data.sample(frac=1).reset_index(drop=True)

        yield batch_x, batch_y
        
            

def get_model():
    model = Sequential()

    #model.add(Reshape(target_shape=(NEW_H, NEW_W, N_CHANNEL), input_shape=(NEW_H, NEW_W)))
    #model.add(AveragePooling2D((2, 2), strides=(2, 2), border_mode='same'))
    ### SOLUTION: Layer 1: Convolutional follwed by Maxpooling.
    #model.add(Conv2D(2, 5, 5, subsample=(2, 2), border_mode='valid'))
    #model.add(Conv2D(2, 3, 3, subsample=(2, 2), border_mode='valid', activation='relu'))
    #model.add(ELU())
    model.add(Conv2D(32, 5, 5, subsample=(1, 1), border_mode='valid', input_shape=(40, 160, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), border_mode='same'))

    ### SOLUTION: Layer 2: Convolutional follwed by Maxpooling.
    model.add(Conv2D(64, 3, 3, subsample=(1, 1), border_mode='valid', activation='relu'))
    #model.add(Dropout(0.4))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), border_mode='same'))
    #model.add(Conv2D(4, 3, 3, subsample=(1, 1), border_mode='valid'))
    #model.add(ELU())
    #model.add(MaxPooling2D((2,2), strides=(2,2), border_mode='valid'))
    #model.add(Dropout(0.25))

    ### SOLUTION: Layer 3: Flatten.
    model.add(Flatten())

    ### SOLUTION: Layer 4: Fully connected with dropout.
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))
    
    ### SOLUTION: Layer 5: Fully connected with a single output since this is a regression problem.
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    return model


if __name__ == '__main__':
    import pandas as pd

    # CONSTANT for training model
    BATCH_SIZE = 32
    NB_EPOCH = 100

    log = pd.read_csv(path.join(BASE_DATA_PATH, 'driving_log.csv'), usecols=[0, 1, 2, 3])

    # generate additional samples
    log = regenerate_sample_log(log)

    # shuffle the data
    log = log.sample(frac=1, random_state=1).reset_index(drop=True)

    # split the whole data into training data and validating data
    split_ratio = 0.8
    split_point = int(log.shape[0]*split_ratio)
    training_log = log.loc[0:split_point - 1]
    validating_log = log.loc[split_point:]

    # determine the samples per epoch and samples for the validation
    samples_per_epoch = (training_log.shape[0]//BATCH_SIZE) * BATCH_SIZE
    nb_val_samples = (validating_log.shape[0]//BATCH_SIZE) * BATCH_SIZE

    # release data from memory
    log = None

    training_generator = get_generator(training_log, batch_size=BATCH_SIZE)
    validating_generator = get_generator(validating_log, batch_size=BATCH_SIZE)

    model = get_model()
    model.summary()

    checkpoint = ModelCheckpoint("model.h5", monitor='val_mean_squared_error', verbose=1, save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_mean_squared_error', min_delta=0.0001, patience=5, verbose=1, mode='min')

    model.fit_generator(generator=training_generator, 
                        validation_data=validating_generator,
                        samples_per_epoch=samples_per_epoch, 
                        nb_epoch=NB_EPOCH, 
                        nb_val_samples=nb_val_samples,
                        callbacks=[checkpoint, early_stop])
    
    print("Training is completed!!")

    #model.save_weights('model.h5')
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    print("The model and trained weight are saved!!")
