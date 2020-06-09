import cv2
import os
import numpy as np
import tensorflow as tf
import pathlib
AUTOTUNE = tf.data.experimental.AUTOTUNE

class Data_loader:

    def __init__(self):

        self.data = []
        self.labels = []

        self.CLASS_NAMES = ['low', 'moderate', 'high']      
        self.BATCH_SIZE = 32
        self.IMG_HEIGHT = 320
        self.IMG_WIDTH = 320

        data_dir = "bark_dataset"
        data_dir = pathlib.Path(data_dir)

        self.list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))
        self.labeled_ds = self.list_ds.map(self.process_path, num_parallel_calls=AUTOTUNE)
        self.train_ds = self.prepare_for_training(self.labeled_ds)


    def get_label(self, file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        return parts[-2] == self.CLASS_NAMES

    def decode_img(self,img):
       # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [self.IMG_HEIGHT, self.IMG_WIDTH])


    def process_path(self,file_path):
        label = self.get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img, label    

    def prepare_for_training(self,ds, cache=True, shuffle_buffer_size=1000):
     # This is a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets that don't
        # fit in memory.
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()

        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        # Repeat forever
        ds = ds.repeat()

        ds = ds.batch(self.BATCH_SIZE)

        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds         
        
     

   

