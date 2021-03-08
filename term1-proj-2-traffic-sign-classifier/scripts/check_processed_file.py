import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load data and show basic info of the image data
training_file = 'train.grey.p'
testing_file = 'test.grey.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)

with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
train_features, train_labels = train['features'], train['labels']
test_features, test_labels = test['features'], test['labels']

print(train_features[0].shape)
print(train_features[0][0][0])


fig = plt.figure(figsize=(2,1))

# Processed Y channel of image
fig.add_subplot(1,2,1)
plt.imshow(train_features[0], cmap='gray')
plt.axis('off')

fig.add_subplot(1,2,2)
plt.imshow(test_features[0], cmap='gray')
plt.axis('off')

plt.show()
