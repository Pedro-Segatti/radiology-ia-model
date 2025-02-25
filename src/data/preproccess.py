import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class PreProccessData:
    def __init__(self, data_dir, target_size=(64, 64), batch_size=32):
        self.data_dir = data_dir
        self.target_size = target_size
        self.batch_size = batch_size
    
    def preprocess_data(self):
        datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
        
        train_gen = datagen.flow_from_directory(
            self.data_dir,
            self.target_size,
            self.batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        val_gen = datagen.flow_from_directory(
            self.data_dir,
            self.target_size,
            self.batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        return train_gen, val_gen
