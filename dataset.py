import os
import math
import random

import pandas as pd
import numpy as np
from PIL import Image

from config import METADATA, SET_FRACTIONS


def load_image_stack(sample):
    '''Return a 3D image stack composed of 7 layers
    3 RGB layers for the satellite image
    3 RGB layers for the terrain image
    and 1 monochrome layer for the streets image
    
    The final stack is (256, 256, 7)'''

    angle = random.randint(-1, 1)*15
    im = Image.open(sample.satellite_path)
    im = im.rotate(angle=angle)
    satellite = np.array(im)

    im = Image.open(sample.terrain_path)
    im = im.rotate(angle=angle)
    terrain = np.array(im)

    im = Image.open(sample.streets_path).convert('L')
    im = im.rotate(angle=angle)
    streets = np.array(im)

    return np.dstack((satellite, terrain, streets))

def layer_means(dataset):
    means = 0
    count = 0
    for images, labels in dataset.batch_generator(10):
        means += np.mean(images, axis=(0, 1, 2))
        count += 1

    return means / count

def get_data_sets(set_fractions=SET_FRACTIONS, augment=True):
    # The callable that will return sets of images to be used in training, validation and testing
    #
    # OUTPUT: a dictionary containing a Dataset object for each set [train, validation, test].

    metadata = pd.read_csv(METADATA)
    # randomly shuffle the rows but ensure that we always get the same random sets
    metadata =  metadata.sample(frac=1, random_state=0).reset_index(drop=True)
    labels = metadata.climbing.values
    num_samples = len(metadata)
    data_sets = {}
    start_idx = 0

    for set_name, set_fraction in set_fractions.items():
        set_size = math.floor(num_samples * set_fraction)
        data_sets[set_name] = Dataset(metadata.iloc[start_idx:start_idx+set_size], augment=augment)
        start_idx += set_size

    return data_sets


class Dataset:
    '''
    A class that stores image paths and labels and provides useful methods
    for delivering batches of training data
    '''

    def __init__(self, metadata, augment=True):
        # batch_idx keeps track of which un-augmented samples have been seen
        self.batch_idx = 0
        # example_idx keeps track of which augmented samples have been seen
        self.example_idx = 0
        self.metadata = metadata
        self.augment = augment
        self.labels = metadata.climbing.values
        self.num_examples = len(metadata)
        if augment:
            # get 8x more samples using 0, 90, 180, 270 degree rotations and 2 flips
            self.num_examples *= 8 * 3
        (self.IMG_X, self.IMG_Y, self.IMG_D) = (256, 256, 7)

    def get_image_batch(self, batch_size, transform=lambda x: x, shuffle=True):
        # return a new batch of images and labels
        if self.batch_idx + batch_size > len(self.metadata):
            self.batch_idx = 0
        if (self.example_idx + batch_size > self.num_examples) and shuffle:
            self.metadata =  self.metadata.sample(frac=1).reset_index(drop=True)

        image_array = np.empty(shape=(batch_size, self.IMG_X, self.IMG_Y, self.IMG_D), dtype=np.float32)
        label_array = np.empty(shape=(batch_size), dtype=np.int32)

        for i, j in zip(range(self.batch_idx, self.batch_idx + batch_size), range(batch_size)):
            # read the image file
            image_array[j, :, :, :] = load_image_stack(self.metadata.iloc[i])
            label_array[j] = self.metadata.iloc[i].climbing
            self.batch_idx += 1
            self.example_idx += 1

        if self.augment:
            image_array = np.rot90(image_array, k=random.randint(0, 3), axes=(1, 2))
            image_array = np.flip(image_array, axis=random.randint(1, 2))

        return transform((image_array, label_array))

    def batch_generator(self, batch_size, transform=lambda x: x, loop=False, shuffle=True):
        '''
        A generator that will yield batches of training data

        transform: an optional function that modifies the raw images and labels returned by get_image_batch
        and returns a new tupel (images, labels)
        '''
        if loop:
            return self._infinite_generator(batch_size, transform, shuffle)
        else:
            return self._finite_generator(batch_size, transform, shuffle)

    def _infinite_generator(self, batch_size, transform, shuffle):
        '''
        Will continue to yield batches of data without end, so don't use it in a list comprehension
        '''
        while True:
            (images, labels) = self.get_image_batch(batch_size, transform, shuffle)
            yield (images, labels)

    def _finite_generator(self, batch_size, transform, shuffle):
        '''
        Will yield batches of data for one epoch, i.e. all samples have been used one time
        '''
        num = 1
        while num * batch_size <= self.num_examples:
            (images, labels) = self.get_image_batch(batch_size, transform, shuffle)
            yield (images, labels)
            num += 1

    def layer_mean_variance(self):
        '''
        Calculate the mean and variance of each layer in the image stack over a complete dataset
        '''

        def update(aggregate, newValue):
            (count, mean, M2) = aggregate
            count = count + 1 
            delta = newValue - mean
            mean = mean + delta / count
            delta2 = newValue - mean
            M2 = M2 + delta * delta2

            return (count, mean, M2)

        count = 0
        mean = np.zeros(shape=self.IMG_D)
        M2 = np.zeros(shape=self.IMG_D)
        for _, sample in self.metadata.iterrows():
            image_stack = load_image_stack(sample)
            means = np.mean(image_stack, axis=(0, 1))
            (count, mean, M2) = update((count, mean, M2), means)

        return (mean, M2/count)
