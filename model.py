import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.src.optimizers import Adam
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Flatten, Dense, Lambda, Cropping2D

# To specify the directory, so we can read and manipulate the image data and cvs files
datadir = "training_set/IMG/"


def load_data():
    col = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
    df = pd.read_csv('training_set/IMG/driving_log.csv', names=col)
    print('total data:', len(df))
    return df


# Balancing DATA
def balance_data(df):
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
    print(len(df.index))
    return df


# Preparing Image and Steering data
def prepare_image_steering(df):
    # image_path = []
    # steering = []
    # for index, row in df.iterrows():
    #     image_path.append(os.path.join(datadir, row["center"].strip()))
    #     steering.append(float(row["steering"]))
    # image_paths_array = np.asarray(image_path)
    # steering_array = np.asarray(steering)
    # return image_paths_array, steering_array

    image_paths_series = df['center'].values
    image_paths_series = np.append(image_paths_series, df['left'].values)
    image_paths_series = np.append(image_paths_series, df['right'].values)
    steerings_series = df['steering'].values
    steerings_series = np.append(steerings_series, df['steering'].values)
    steerings_series = np.append(steerings_series, df['steering'].values)

    return image_paths_series, steerings_series


# def read_image(img_path):
# To read the image paths we provided and store the actual image it contains:
# read_img = cv2.imread(img_path)
# return read_img
# for images, ste in zip(image_paths_series, steerings_series):
#     aug_imgs = []
#     for img in images:
#         image_open = cv2.imread(img)
#         # Flipping the image
#         aug_imgs.append((cv2.flip(image_open, 1)))
#     np.append(image_paths_series, aug_imgs)
#     # Flipping Steering angle
#     np.append(steerings_series, ste * -1.0)


def img_preprocess(img_path):
    # To read the image paths we provided and store the actual image it contains:
    img = cv2.imread(img_path)
    # NVidia Model requires to change our image color-space from RGB to YUV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # Applying 3x3-Kernel GaussianBlur to smooth the image and reduce noise
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def build_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255) - 0.5, input_shape=(160, 320, 3)))

    # Cropping the top 60 pixels and the bottom 25 pixels from the image
    model.add(Cropping2D(cropping=((60, 25), (0, 0))))

    # model.add(Lambda(p))
    # The layers
    model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(units=100, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=1))
    optimizer = Adam(lr=0.0004)
    model.compile(loss='mse', optimizer=optimizer)
    return model


def train_model(model, x_data, y_data):
    model.fit(x_data, y_data, batch_size=100, epochs=20, verbose=1, validation_split=0.2, shuffle=True)
    model.save('model.h5')


def main():
    dataframe = load_data()
    dataframe = balance_data(dataframe)
    image_paths, steering_values = prepare_image_steering(dataframe)
    # read_images = np.array(list(map(read_image, image_paths)))
    x_data = np.array(list(map(img_preprocess, image_paths)))
    y_data = steering_values
    model = build_model()
    train_model(model, x_data, y_data)


if __name__ == '__main__':
    main()
