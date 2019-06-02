import pandas as pd
from sklearn.model_selection import train_test_split
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import shutil
import numpy as np
from PIL import Image


# parameters
LOAD_IMAGES_PATH = '../data/original_data/HAM'
EXTENSION = '.jpg'
METADATA_PATH = '../data/original_data/HAM10000_metadata.tab'
SAVE_IMAGES_PATH = '../data'
SAVE_STATS_PATH = '../data/stats'
SEPARATOR = '\t'
RANDOM_STATE = 42
TRAIN_SIZE = 0.75
HEIGHT = 450
WIDTH = 600
NEW_HEIGHT = HEIGHT // 2
NEW_WIDTH = WIDTH // 2
BATCH_SIZE = 16
TOTAL_NUMBER_OF_IMAGES_PER_CLASS = 4500


# read metadata
df = pd.read_csv(METADATA_PATH, sep=SEPARATOR, engine='python')
# divide into validation set and rest
_, val = train_test_split(df, train_size=TRAIN_SIZE, shuffle=True, random_state=RANDOM_STATE)
# get train set without records from validation including naturally augmented images
train = df[~df['lesion_id'].isin(val['lesion_id'].tolist())]

print('Original training set - class distribution')
print(train['dx'].value_counts())
print('Original validation set - class distribution')
print(val['dx'].value_counts())

# get column with unique classes
classes = set(df['dx'])

# make directory for training and validation data
train_and_val_data_path = os.path.join(SAVE_IMAGES_PATH, 'train_and_val_data')
os.mkdir(train_and_val_data_path)

# make directory for training data and copy files there
train_path = os.path.join(train_and_val_data_path, 'train_data')
os.mkdir(train_path)
for train_file in train['image_id'].values:
    shutil.copyfile(os.path.join(LOAD_IMAGES_PATH, train_file + EXTENSION), os.path.join(train_path, train_file + EXTENSION))

# make directory for validation data and copy files there
val_path = os.path.join(train_and_val_data_path, 'val_data')
os.mkdir(val_path)
for val_file in val['image_id'].values:
    shutil.copyfile(os.path.join(LOAD_IMAGES_PATH, val_file + EXTENSION), os.path.join(val_path, val_file + EXTENSION))

# make directory for processed training and validation data
processed_train_and_val_data_path = os.path.join(train_and_val_data_path, 'processed_train_and_val_data')
os.mkdir(processed_train_and_val_data_path)

# make directory for processed training data
processed_train_data_path = os.path.join(processed_train_and_val_data_path, 'train_data')
os.mkdir(processed_train_data_path)

# define augmented generator for training data
train_images_data_generator = ImageDataGenerator(rotation_range=180,
                                                 width_shift_range=0.2,
                                                 height_shift_range=0.2,
                                                 zoom_range=0.2,
                                                 shear_range=0.2,
                                                 horizontal_flip=True,
                                                 vertical_flip=True,
                                                 fill_mode='nearest')
# make directory for processed validation data
processed_val_data_path = os.path.join(processed_train_and_val_data_path, 'val_data')
os.mkdir(processed_val_data_path)
for _class in classes:
    print('Processing class {} for training set'.format(_class))

    # make temporary directory for data
    data_tmp_path = os.path.join(train_and_val_data_path, 'data_tmp')
    os.mkdir(data_tmp_path)

    # make subdirectory for temporary class in temporary directory
    data_tmp_class_path = os.path.join(data_tmp_path, _class)
    os.mkdir(data_tmp_class_path)

    # make directory for class in processed training data directory
    processed_train_data_class_path = os.path.join(processed_train_data_path, _class)
    os.mkdir(processed_train_data_class_path)

    # get image_id of training images with specific class
    train_images_original = (train.loc[train['dx'] == _class])['image_id'].values
    train_images_original_size = len(train_images_original)

    for train_file in train_images_original:
        # copy images from training data directory to temporary directory
        shutil.copyfile(os.path.join(train_path, train_file + EXTENSION), os.path.join(data_tmp_class_path, train_file + EXTENSION))

    # augment images from temporary directory
    augmented_images_gen = train_images_data_generator.flow_from_directory(data_tmp_path,
                                                                           save_to_dir=processed_train_data_class_path,
                                                                           save_format='jpg',
                                                                           target_size=(NEW_HEIGHT, NEW_WIDTH),
                                                                           batch_size=BATCH_SIZE)

    # define number of additional (augmented) images in batches
    diff = TOTAL_NUMBER_OF_IMAGES_PER_CLASS - train_images_original_size
    if diff > 0:
        print('DIFF: {}'.format(diff))
        number_of_batches = int(np.ceil(train_images_original_size / BATCH_SIZE) * np.floor(diff / train_images_original_size) +
                                np.ceil((diff % train_images_original_size) / BATCH_SIZE))
        for _ in range(number_of_batches):
            _, _ = next(augmented_images_gen)
    else:
        print('diff <= 0 !')

    # resize original training images and copy them to augmented images
    for train_file in train_images_original:
        input_path = os.path.join(train_path, train_file + EXTENSION)
        save_path = os.path.join(processed_train_data_class_path, train_file + EXTENSION)
        resized_train_image = Image.fromarray(np.asarray(Image.open(input_path).resize((NEW_WIDTH, NEW_HEIGHT))),
                                              mode='RGB')
        resized_train_image.save(save_path)

    # remove temporary tree of directories
    shutil.rmtree(data_tmp_path)

    print('Processing class {} for validation set'.format(_class))
    # make directory for class in processed validation data directory
    processed_val_data_class_path = os.path.join(processed_val_data_path, _class)
    os.mkdir(processed_val_data_class_path)

    # get image_id of validation images with specific class
    val_images_original = (val.loc[val['dx'] == _class])['image_id'].values

    # resize original validation images and copy them to class subdirectory in processed validation data directory
    for val_file in val_images_original:
        input_path = os.path.join(val_path, val_file + EXTENSION)
        save_path = os.path.join(processed_val_data_class_path, val_file + EXTENSION)
        resized_val_image = Image.fromarray(np.asarray(Image.open(input_path).resize((NEW_WIDTH, NEW_HEIGHT))),
                                            mode='RGB')
        resized_val_image.save(save_path)
