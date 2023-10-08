import numpy as np

import tensorflow as tf
from tensorflow import keras


# TODO 1: Create a your own custom Dataset here


class DummyDataset:
    def __init__(self, paths, transform=None, is_predict=False):
        super().__init__()
        self.paths = paths
        self.transform = transform
        self.is_predict = is_predict

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        image = tf.keras.preprocessing.image.load_img(path)
        item = tf.keras.preprocessing.image.img_to_array(image)
        if self.transform:
            item = self.transform(item)
        if self.is_predict:
            return item
        target = True
        return item, target


class DummyDataGenerator(keras.utils.Sequence):
    def __init__(self, dataset, batch_size, shuffle=False):
        super().__init__()
        self.dataset = dataset
        self.indexes = np.arange(len(self.dataset))
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return len(self.paths) // self.batch_size

    def __getitem__(self, i):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[i * self.batch_size: (i + 1) * self.batch_size]

        # Find list of IDs
        return [self.dataset[k] for k in indexes]

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)
