from argparse import ArgumentParser
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dense, Dropout, Input
from tensorflow.python.estimator.export.export import TensorServingInputReceiver
from tensorflow.python.keras.applications.xception import Xception, preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.keras.estimator import model_to_estimator
from functools import partial
import numpy as np
import csv
import os


def build_model(dropout, learning_rate):
    input = Input(shape=(HEIGHT, WIDTH, 3))
    base_model = Xception(input_tensor=input, weights='imagenet', include_top=False, pooling='avg')
    for layer in base_model.layers:
        layer.trainable = True
    x = base_model.output
    x = Dropout(dropout)(x)
    x = Dense(NUMBER_OF_CLASSES, activation='softmax')(x)
    model = Model(inputs=input, outputs=x)
    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


def serving_input_fn():
    inputs = tf.placeholder(tf.string, shape=[None])
    images = tf.map_fn(partial(tf.image.decode_image, channels=DEPTH), inputs, dtype=tf.uint8)
    images = tf.reshape(images, [-1, HEIGHT, WIDTH, DEPTH])
    images = preprocess_input(tf.cast(images, tf.float32))
    return TensorServingInputReceiver(features=images, receiver_tensors=inputs)


def main(**args):
    DATA_DIR = args['data_dir']
    OUTPUT_DIR = args['output_dir']
    KERAS_SAVE_PERIOD = args['keras_save_period']
    TRAIN_BATCH_SIZE = int(args['train_batch_size'])
    EPOCHS = int(args['epochs'])
    STEPS_PER_EPOCH = int(args['steps_per_epoch'])
    VALIDATION_STEPS = int(args['validation_steps'])
    VAL_BATCH_SIZE = int(args['val_batch_size'])
    DROPOUT_RATE = float(args['dropout_rate'])
    LEARNING_RATE = float(args['learning_rate'])
    PLATEAU_FACTOR = float(args['plateau_factor'])
    PLATEAU_PATIENT = int(args['plateau_patient'])
    PLATEAU_MIN_LEARNING_RATE = float(args['plateau_min_learning_rate'])
    global NUMBER_OF_CLASSES, HEIGHT, WIDTH, DEPTH
    NUMBER_OF_CLASSES = int(args['number_of_classes'])
    WIDTH = int(args['width'])
    HEIGHT = int(args['height'])
    DEPTH = int(args['depth'])
    TRAIN_PATH = os.path.join(DATA_DIR, 'train_data')
    VAL_PATH = os.path.join(DATA_DIR, 'val_data')
    KERAS_MODEL_PATH = os.path.join(OUTPUT_DIR, 'keras_model{epoch:02d}.h5')
    ESTIMATOR_PATH = os.path.join(OUTPUT_DIR, 'estimator')
    EXPORT_MODEL_PATH = os.path.join(OUTPUT_DIR, 'export')
    TENSORBOARD_PATH = os.path.join(OUTPUT_DIR, 'logs')
    TRAIN_RESULTS_PATH = os.path.join(OUTPUT_DIR, 'train_results')
    VAL_RESULTS_PATH = os.path.join(OUTPUT_DIR, 'val_results')

    model = build_model(DROPOUT_RATE, LEARNING_RATE)

    data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_gen = data_gen.flow_from_directory(VAL_PATH,
                                           target_size=(HEIGHT, WIDTH),
                                           color_mode='rgb',
                                           batch_size=VAL_BATCH_SIZE,
                                           shuffle=False)

    train_images_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input,
                                                     rotation_range=180,
                                                     width_shift_range=0.2,
                                                     height_shift_range=0.2,
                                                     zoom_range=0.2,
                                                     shear_range=0.2,
                                                     horizontal_flip=True,
                                                     vertical_flip=True,
                                                     fill_mode='nearest')
    augmented_images_gen = train_images_data_generator.flow_from_directory(TRAIN_PATH,
                                                                           shuffle=True,
                                                                           target_size=(HEIGHT, WIDTH),
                                                                           batch_size=TRAIN_BATCH_SIZE)

    model_checkpoint = ModelCheckpoint(KERAS_MODEL_PATH, period=KERAS_SAVE_PERIOD, save_weights_only=True)
    reduce_plateau = ReduceLROnPlateau(monitor='val_categorical_accuracy',
                                       factor=PLATEAU_FACTOR,
                                       patience=PLATEAU_PATIENT,
                                       verbose=1,
                                       mode='max',
                                       min_lr=PLATEAU_MIN_LEARNING_RATE)
    tensorboard = TensorBoard(log_dir=TENSORBOARD_PATH, histogram_freq=0, write_graph=True, write_images=False)

    model.fit_generator(augmented_images_gen,
                        validation_data=val_gen,
                        epochs=EPOCHS,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_steps=VALIDATION_STEPS,
                        callbacks=[model_checkpoint, reduce_plateau, tensorboard])

    train_val_gen = data_gen.flow_from_directory(TRAIN_PATH,
                                                 target_size=(HEIGHT, WIDTH),
                                                 color_mode='rgb',
                                                 batch_size=VAL_BATCH_SIZE,
                                                 shuffle=False)

    y_train = train_val_gen.classes
    y_train_ids = train_val_gen.filenames
    # we are assuming that STEPS_PER_EPOCH represents iteration over whole training set
    y_train_pred = model.predict_generator(train_val_gen,
                                           steps=int(np.ceil(STEPS_PER_EPOCH * TRAIN_BATCH_SIZE / VAL_BATCH_SIZE)))

    # we are assuming that VALIDATION_STEPS represents iteration over whole training set
    val_gen.reset()
    y_val = val_gen.classes
    y_val_ids = val_gen.filenames
    y_val_pred = model.predict_generator(val_gen, steps=VALIDATION_STEPS)


    with open(TRAIN_RESULTS_PATH, mode='w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow(['nr', 'image_id', 'true', 'predicted', 'probabilities'])
        for i, example in enumerate(y_train_pred):
            csv_writer.writerow([i + 1, y_train_ids[i], y_train[i], np.argmax(example), list(example)])

    with open(VAL_RESULTS_PATH, mode='w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow(['nr', 'image_id', 'true', 'predicted', 'probabilities'])
        for i, example in enumerate(y_val_pred):
            csv_writer.writerow([i + 1, y_val_ids[i], y_val[i], np.argmax(example), list(example)])

    estimator = model_to_estimator(keras_model=model, model_dir=ESTIMATOR_PATH)
    # https://github.com/tensorflow/tensorflow/issues/26178
    estimator._model_dir = os.path.join(ESTIMATOR_PATH, 'keras')
    estimator.export_savedmodel(EXPORT_MODEL_PATH, serving_input_receiver_fn=serving_input_fn)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', help='Data directory', type=str, required=True)
    parser.add_argument('--output_dir', help='Output directory', type=str, required=True)
    parser.add_argument('--keras_save_period', help='Period with which we save Keras models', type=int, required=True)
    parser.add_argument('--train_batch_size', help='Train batch size', type=int, required=True)
    parser.add_argument('--epochs', help='Number of epochs', type=int, required=True)
    parser.add_argument('--steps_per_epoch', help='Number of training steps per epoch', type=int, required=True)
    parser.add_argument('--val_batch_size', help='Validation batch size', type=int, required=True)
    parser.add_argument('--validation_steps', help='Validation steps per epoch', type=int, required=True)
    parser.add_argument('--dropout_rate', help='Dropout rate in fully connected layers', type=float, required=True)
    parser.add_argument('--learning_rate', help='Learning rate', type=float, required=True)
    parser.add_argument('--plateau_factor', help='Plateau factor for decreasing learning rate', type=float, required=True)
    parser.add_argument('--plateau_patient', help='How many epochs with no change in observed metric before we trigger learning rate decreasing',
                        type=int, required=True)
    parser.add_argument('--plateau_min_learning_rate', help='Plateau minimal learning rate', type=float, required=True)
    parser.add_argument('--number_of_classes', help='Number of classes', type=int, required=True)
    parser.add_argument('--width', help='Width of images', type=int, required=True)
    parser.add_argument('--height', help='Height of images', type=int, required=True)
    parser.add_argument('--depth', help='Depth of images', type=int, required=True)
    args = parser.parse_args()
    arguments = args.__dict__
    main(**arguments)
