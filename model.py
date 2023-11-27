import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import ntpath

# To specify the directory, so we can read and manipulate the image data and cvs files
datadir = "training_set/IMG/"

col = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
df = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names=col)


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail


df['center'] = df['center'].apply(path_leaf)
df['left'] = df['left'].apply(path_leaf)
df['right'] = df['right'].apply(path_leaf)

print('total data:', len(df))

# Balancing DATA
# To flatten and cut off the steering values where sum exceeds 250 and make it more uniform
num_bins = 25

# Before Removal
hist, bins = np.histogram(df['steering'], num_bins)
center = (bins[:-1] + bins[1:]) * 0.5
plt.bar(x=center, height=hist, width=0.05)
plt.savefig('hist1.png')

samples_per_bin = 250
remove_list = []
for j in range(num_bins):
    list_ = []
    for i in range(df.count().iloc[0]):
        # If the steering angle falls in between two bins, then it belongs to the interval j
        if bins[j] <= df['steering'][i] <= bins[j + 1]:
            list_.append(i)
            # Eventually, this list will contain all the steering numbers from a specific bin.
            # Because our threshold in this project is max 250 steering numbers per bin,
            # we need to reject the exceeding ones, and, because the numbers are stored in an array in order,
            # we need to shuffle first (if we just reject the last ones,
            # we may be rejecting information from the end of our track which is bad for our model to
            # predict how to drive properly on the end of the track)
    list_ = shuffle(list_)
    list_ = list_[samples_per_bin:]
    remove_list.extend(list_)

print('removed:', len(remove_list))
df.drop(df.index[remove_list], inplace=True)
print('remaining:', len(df))

# After Removal
hist, bins = np.histogram(df['steering'], num_bins)
plt.bar(x=center, height=hist, width=0.05)
plt.savefig('hist2.png')


# Preparing Image and Steering data
def prepare_image_steering():
    image_path = []
    steering = []
    for index, row in df.iterrows():
        image_path.append(os.path.join(datadir, row["center"].strip()))
        steering.append(float(row["steering"]))
    image_paths_array = np.asarray(image_path)
    steering_array = np.asarray(steering)
    return image_paths_array, steering_array


image_paths, steerings = prepare_image_steering()
