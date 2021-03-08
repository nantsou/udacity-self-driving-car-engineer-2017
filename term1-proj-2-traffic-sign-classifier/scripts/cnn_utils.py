import tensorflow as tf

# create the weights for cnn
def create_weights(shape, mu=0, sigam=0.01):
    return tf.Variable(tf.truncated_normal(shape=shape, mean=mu, stddev=sigma))

# create the bias for cnn
def create_biases(length):
    return tf.Variable(tf.constant(0.0, shape=[length]))

# create conv layer
def create_conv_layer(input, depth, filter_size, n_filters, strides, padding, use_pooling=True):
    
    # Shape params of filter's Weights for the convolution.
    shape = [filter_size, filter_size, depth, n_filters]

    # Length of filter's bias for the convolution.
    length = n_filters

    weights = create_weights(shape)
    biases = create_biases(length)

    layer = tf.nn.conv2d(input, weights, strides=strides, padding=padding)
    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    
    # activation funciton
    layer = tf.nn.relu(layer)
    
    return layer, weights

# create flattern layer
def create_flattern_layer(layer):
    # get shape of input layer
    layer_shpae = layer.get_shape()

    # The number of features is: img_h * img_w * depth
    n_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num of image, num. of features].
    flat_layer = tf.reshape(layer, [-1, n_features])

    return flat_layer, n_features

# create fully connected layer
def create_fully_connected_layer(input, n_input, n_output, use_relu=True):
    
    # shape for creating weights
    shape = [n_input, n_output]
    # lenght for creating biases
    length = n_output
    
    # create weights
    weights = create_weights(shape=shape)
    # create biases
    biases = create_biases(length=length)

    # create the fully connected layer by matrix multiplication
    layer = tf.matmul(input, weights) + biases

    # process relu
    if use_relu:
        layer = tf.nn.relu(layer)
    
    return layer