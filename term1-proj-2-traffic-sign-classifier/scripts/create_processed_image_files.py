import os
import pickle
import numpy as np
from image_processor import generate_additional_image_data, to_grey_processor

# Load data and show basic info of the image data
training_file = '../data/train.p'
testing_file = '../data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
train_features, train_labels = train['features'], train['labels']
test_features, test_labels = test['features'], test['labels']

# Save processed image dataset
def save_file(file_name, features, labels):
    try:
        with open(file_name, 'wb') as pfile:
            pickle.dump(
                {
                    'features': features,
                    'labels': labels,
                },
                pfile, 
                pickle.HIGHEST_PROTOCOL
            )
    except Exception as e:
        print('Unable to save image data to {0}: {1}'.format(file_name, e))

if __name__ == "__main__":
    processed_train_file = '../data/train.grey.p'
    processed_test_file = '../data/test.grey.p'

    if not os.path.isfile(processed_train_file):
        # Preprocess image data
        ## Add data image
        train_features, train_labels = generate_additional_image_data(train_features, train_labels)
        ## Make image data grey
        train_features = to_grey_processor(train_features)
        test_features = to_grey_processor(test_features)
        
        # save image dataset into files
        save_file(processed_train_file, train_features, train_labels)
        save_file(processed_test_file, test_features, test_labels)