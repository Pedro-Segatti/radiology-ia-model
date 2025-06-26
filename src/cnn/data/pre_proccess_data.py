import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class PreProccessData:
    def __init__(self, data_dir, target_size=(256, 256, 1), batch_size=2):
        self.data_dir = data_dir
        self.target_size = target_size
        self.batch_size = batch_size

    def preprocess_data(self):
        datagen = ImageDataGenerator(rescale=1.0 / 255)

        train_gen = datagen.flow_from_directory(
            directory=os.path.join(self.data_dir, "train"),
            target_size=self.target_size[:2],
            batch_size=self.batch_size,
            class_mode="categorical",
            color_mode="grayscale",
            shuffle=True
        )

        val_gen = datagen.flow_from_directory(
            directory=os.path.join(self.data_dir, "validation"),
            target_size=self.target_size[:2],
            batch_size=self.batch_size,
            class_mode="categorical",
            color_mode="grayscale",
            shuffle=False
        )

        return train_gen, val_gen
