import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, activations, metrics
import tensorflow.keras as keras
from sklearn.metrics import jaccard_score, f1_score, accuracy_score, precision_score, recall_score
import numpy as np


def parse_function(img_fname, label_fname):
    image_string = tf.io.read_file(img_fname)
    label_string = tf.io.read_file(label_fname)
    image = tf.io.decode_png(image_string, channels=3)
    label = tf.io.decode_png(label_string, channels=3)
        
    # convert values to float and in range [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.float32) / 255.0
    
    # create single channel {0, 1} mask from label img
    label = tf.image.rgb_to_grayscale(label)
    label = tf.where(label > 0.0, 1.0, 0.0)

    image = tf.image.resize(image, [256, 256])
    label = tf.image.resize(label, [256, 256])
    
    return image, label


def filter_function(image, label):
    # exclude image, label pairs where less than 1% or more than 90% is greenspace
    pixel_count = tf.cast(tf.math.multiply(label.shape[0], label.shape[1]), dtype=tf.float32)
    min_thresh = tf.math.multiply(pixel_count, tf.constant([0.01])) < tf.math.reduce_sum(label)
    max_thresh = tf.math.reduce_sum(label) < tf.math.multiply(pixel_count, tf.constant([0.9]))
    return tf.squeeze(tf.math.logical_and(min_thresh, max_thresh))


def augmentation_function(image, label, seed=42):
    # orientation augments
    if tf.random.uniform(shape=[]) < 0.5:
        image = tf.image.flip_up_down(image)
        label = tf.image.flip_up_down(label)
    if tf.random.uniform(shape=[]) < 0.5:
        image = tf.image.flip_left_right(image)
        label = tf.image.flip_left_right(label)
        
    # color augments
    image = tf.image.random_brightness(image, 0.1, seed)
    image = tf.image.random_contrast(image, 0.8, 1.2, seed)
    image = tf.image.random_saturation(image, 0.8, 2, seed)
    image = tf.image.random_hue(image, 0.15, seed)
    # sharpness?

    # make sure the image and label is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    label = tf.clip_by_value(label, 0.0, 1.0)
    
    # finally one hot encode label
    # label = tf.squeeze(tf.one_hot(tf.cast(label, tf.uint8), depth=2), axis=2)

    return image, label


def data_pipeline(imgs, labels, batch_size, train_mode=True, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices((imgs, labels))
    if train_mode:
        dataset = dataset.shuffle(len(imgs))
    dataset = dataset.map(parse_function, num_parallel_calls=os.cpu_count())
    dataset = dataset.filter(filter_function)
    if augment:
        dataset = dataset.map(augmentation_function, num_parallel_calls=os.cpu_count())
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(5)  # prefetch 5 batches into memory
    
    return dataset


def plot_history(history, metric='f1-score'):
    metric_score = history.history[metric]
    val_metric_score = history.history['val_'+metric]
    epochs = range(1, len(metric_score) + 1)
    plt.plot(epochs, metric_score, 'y', label=f'Training {metric}')
    plt.plot(epochs, val_metric_score, 'r', label=f'Validation {metric}')
    plt.title(f'Training and Validation {metric}')
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    plt.show()


def evaluate_baseline(model, pipeline, log_thresh=0.5):
    jaccards = []
    f1s = []
    accuracies = []
    precisions = []
    recalls = []
    for batch in pipeline:
        resize_shape = batch[0].shape[0]*256*256
        X_batch = tf.reshape(batch[0], (resize_shape, 3)).numpy()
        y_batch = tf.squeeze(tf.reshape(batch[1], (resize_shape, 1))).numpy()
        pred = np.where(model.predict_proba(X_batch)[:, 1] > log_thresh, 1.0, 0.0)
        jaccards.append(jaccard_score(y_batch, pred))
        f1s.append(f1_score(y_batch, pred))
        accuracies.append(accuracy_score(y_batch, pred))
        precisions.append(precision_score(y_batch, pred, zero_division=0))
        recalls.append(recall_score(y_batch, pred))
    return [np.mean(accuracies), np.mean(jaccards), np.mean(f1s), np.mean(precisions), np.mean(recalls)]
