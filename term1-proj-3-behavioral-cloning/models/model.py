# -*- coding: utf-8 -*-

from keras.layers import Convolution2D as Conv2D, Dropout, Flatten, Dense, MaxPooling2D, ELU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential

def get_model():
    model = Sequential()
    ### SOLUTION: Layer 1: Convolutional with 4x4 patch, 16 features and (4, 4) stride of 'same' border
    model.add(Conv2D(16, 4, 4, subsample=(4, 4), input_shape=(16, 64, 1), border_mode='same'))
    ###@ Activation function: Exponential linear unit
    model.add(ELU())
    
    ### SOLUTION: Layer 2: Maxpooling with 2x2 patch and (1, 2) stride of 'valid' border
    model.add(MaxPooling2D((2, 2), strides=(1, 2), border_mode='valid'))

    ### SOLUTION: Layer 3: Convolutional with 2x2 patch, 32 feature and (2, 2) stride of 'same' border
    model.add(Conv2D(32, 2, 2, subsample=(2, 2), border_mode='same'))
    #### Activation function: Exponential linear unit
    model.add(ELU())
    #### Droupout 50%
    model.add(Dropout(0.5))
    
    ### SOLUTION: Layer 4: Maxpooling with 2x2 patch and (1, 2) stride of valid border
    model.add(MaxPooling2D((2,2), strides=(1, 2), border_mode='valid'))

    ### SOLUTION: Layer 5: Flatten
    model.add(Flatten())

    ### SOLUTION: Layer 6: Fully connected with output 512 feature
    model.add(Dense(512))
    #### Dropout 50%
    model.add(Dropout(0.5))
    #### Activation function: Exponential linear unit
    model.add(ELU())

    ### SOLUTION: Layer 7: Fully connected with a single output since this is a regression problem.
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model


if __name__ == '__main__':
    import numpy as np
    from os import path
    import argparse

    parser = argparse.ArgumentParser(description='Training model.')
    parser.add_argument(
        '--batch', type=int, default=128, help='batch size.')
    parser.add_argument(
        '--epoch', type=int, default=100, help='number of epochs.')
    args = parser.parse_args()

    # load training data with default file name
    x = np.load('x.data.train.npy')
    y = np.load('y.data.train.npy')

    # get model
    model = get_model()
    # show model architecture
    model.summary()

    # export model architecture
    model_json = model.to_json()
    with open("model.json", "w") as f:
        f.write(model_json)
    checkpoint = ModelCheckpoint(
        "model.h5", 
        monitor='val_mean_squared_error', 
        verbose=1, 
        save_best_only=True, 
        mode='min')
    early_stop = EarlyStopping(
        monitor='val_mean_squared_error', 
        min_delta=0.0001, 
        patience=4, 
        verbose=1, 
        mode='min')
    history = model.fit(
        x, 
        y, 
        batch_size=args.batch, 
        nb_epoch=args.epoch, 
        verbose=1, 
        callbacks=[checkpoint, early_stop], 
        validation_split=0.2, 
        shuffle=True)
    
