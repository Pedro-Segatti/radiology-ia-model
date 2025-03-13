from tensorflow.keras.preprocessing.image import ImageDataGenerator


class AugmentData:
    def augment_data(self):
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode="nearest"
        )

        return datagen
