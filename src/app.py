import os

import numpy as np
import tensorflow as tf

from models.radiology_model import RadiologyModel
from train.train import Train


def load_data(self):
        """
        Carrega e processa os dados de imagem.

        :return: Tupla contendo os dados de treinamento e teste.
        """
        # Inicializa listas para imagens e rótulos
        images = []
        labels = []

        # Carrega imagens e rótulos
        for label, class_dir in enumerate(os.listdir(self.data_dir)):
            class_path = os.path.join(self.data_dir, class_dir)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=self.input_shape[:2])
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    images.append(img_array)
                    labels.append(label)

        # Converte listas para arrays numpy
        images = np.array(images, dtype='float32') / 255.0  # Normalização
        labels = to_categorical(labels, num_classes=self.num_classes)

        # Divide os dados em treinamento e teste
        return train_test_split(images, labels, test_size=self.test_size, random_state=42)

radiology_model = RadiologyModel((128, 128, 3), 2)
radiology_model.build_model()
radiology_model.compile_model(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
radiology_model = radiology_model.get_model()

Train.train_model()


main = Main(data_dir="path/to/radiology/images", input_shape=(128, 128, 3), num_classes=2)
main.train_model()
main.evaluate_model()
print(main.predict("path/to/single/image.jpg"))