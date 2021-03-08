import os
import pickle
import numpy as np
from sklearn.utils import shuffle

from image_processor import (generate_additional_image_data, 
                             to_grey_processor, 
                             flatten_dataset, 
                             one_hot_encoding)
from cnn_utils import (create_conv_layer, 
                       create_flattern_layer, 
                       create_fully_connected_layer)

####################
### loading data ###
####################

# Load data and show basic info of the image data
training_file = '../train.p'
testing_file = '../test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
train_features, train_labels = train['features'], train['labels']
test_features, test_labels = test['features'], test['labels']

n_train = len(train_features)
n_test = len(test_features)
image_shape = '{0}x{1}'.format(len(train_features[0]), len(train_features[0][0]))
n_classes = max(train_labels) + 1

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

#############################
### preprocess image data ###
#############################

# Add additional image data
train_features, train_labels = generate_additional_image_data(train_features, train_labels)

## Shuffle the training dataset to avoid the bad influence caused by the order of original images and rotated images.
train_features, train_labels = shuffle(train_features, train_labels)

# Make color image data grey
train_features = to_grey_processor(train_features)
test_features = to_grey_processor(test_features)

# Flatten image data
train_features = flatten_dataset(train_features)
test_features = flatten_dataset(test_features)

train_labels = one_hot_encoding(train_labels)
test_labels = one_hot_encoding(test_labels)

# Genreate dataset for training and validation
from sklearn.model_selection import train_test_split
train_features, valid_features, train_labels, valid_labels = train_test_split(train_features,
                                                                              train_labels,
                                                                              test_size=0.2,
                                                                              random_state=741213)

########################
### CNN Architecture ###
########################
import tensorflow as tf

# define basic parameters
img_size = 32
flat_img_size = img_size*img_size
depth = 1 # 1 channel of the image (grey)
n_class = n_classes # It should be 43 in this project

# create cnn network architecture
## define parameters for each layer
# Convolutional Layer 1.
filter_size_1 = 5          # Convolution filters are 5 x 5 pixels.
n_filters_1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size_2 = 5          # Convolution filters are 5 x 5 pixels.
n_filters_2 = 36         # There are 36 of these filters.

# fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

# build network
def cnn_network(x):
    strides = [1,1,1,1]

    # SOLUTION: Layer 1: Convolutional. Input = 1x1024(transfer to 32x32x1 inside).
    conv_1, weights_1 = create_conv_layer(input=x, 
                                         depth=depth, 
                                         filter_size=filter_size_1, 
                                         n_filters=n_filters_1, 
                                         strides=strides, 
                                         padding='SAME')

    # SOLUTION: Layer 2: Convolutional.
    conv_2, weights_2 = create_conv_layer(input=conv_1,
                                           depth=n_filter_1,
                                           filter_size=filter_size_2,
                                           n_filters=n_filters_2,
                                           strides=strides,
                                           padding='SAME')
    
    # SOLUTION: layer 3: Flattern.
    flat_layer, n_features = create_flattern_layer(layer=conv_2)

    # SOLUTION: Layer 4: Fully connected.
    fc_1 = create_fully_connected_layer(input=flat_layer,
                                           n_input=n_features,
                                           n_output=fc_size)

    # SOLUTION: Layer 5: Fully connected.
    logist = create_fully_connected_layer(input=fc_1,
                                           n_input=fc_size,
                                           n_output=n_class,
                                           use_relu=False)

    return logist


# define tensorflow placeholder for the input dataset
x = tf.placeholder(tf.float32, shape=[None, flat_img_size]) 
x_input = tf.reshape(x, [-1, img_size, img_size, depth])

# define tensorflow placeholder for the output dataset
y = tf.placeholder(tf.float32, [None, n_class])
y_cls = tf.argmax(y, dimension=1)

# Training pipeline
BATCH_SIZE = 100
EPOCHS = 100

rate = 0.001 # As the instruction said, 0.001 is a good default value for the learning rate.

logist = cnn_network(x_input)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels_true)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

# Model evaluation
y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_cls)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(x_data, y_data):
    n_samples = len(x_data)
    total_accuracy = 0
    session = tf.get_default_session()
    for offset in range(0, n_samples, BATCH_SIZE):
        x_batch, y_batch = x_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = session.run(accuracy_operation, feed_dict={x: x_batch, y: y_batch})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / n_samples

# Train Model
with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    n_samples = len(train_features)

    print('Start training model...\n')
    for i in range(EPOCHS):
        train_features, train_labels = shuffle(train_features, train_labels)
        for offset in range(0, n_samples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_train_features, batch_train_labels = train_features[offset:end], train_labels[offset:end]
            session.run(training_operation, feed_dict={x: batch_train_features, y:batch_train_labels})
        
        validation_accuracy = evaluate(valid_features, valid_labels)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}\n".format(validation_accuracy))

    try:
        saver
    except NameError:
        saver = tf.train.Saver()
    
    saver.save(session, 'cnn_network')
    print("Model saved")

# Evaluate the Model
with tf.Session() as session:
    loader = tf.train.import_meta_graph('cnn_network.meta')
    loader.restore(sess, tf.train.latest_checkpoint('./'))

    test_accuracy = evaluate(test_features, test_labels)
    