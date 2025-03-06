import os

import numpy as np
import tensorflow as tf
from data.augment import AugmentData
from data.preproccess import PreProccessData
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


class Train:
    def __init__(
        self, model, data_dir, input_shape, num_classes, epochs=10, batch_size=32
    ):
        self.model = model
        self.data_dir = data_dir
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size

        self.x_train, self.x_test, self.y_train, self.y_test = self.load_data()

    def load_data(self):
        """
        Carrega e processa os dados de imagem.

        :return: Tupla contendo os dados de treinamento e teste.
        """
        # Inicializa listas para imagens e rótulos
        images = []
        labels = []

        # Carrega imagens e rótulos
        for class_dir in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_dir)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    img = tf.keras.preprocessing.image.load_img(
                        img_path, target_size=(128, 128, 3)
                    )
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    images.append(img_array)
                    labels.append(class_dir)  # Usa o nome da pasta como label
                    print("Imagem carregada:", img_name, "| Classe:", class_dir)

        # Converte listas para arrays numpy
        images = np.array(images, dtype="float32") / 255.0  # Normalização

        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        self.model.classes_ = label_encoder.classes_

        # Converte os rótulos numéricos para one-hot encoding
        num_classes = len(label_encoder.classes_)

        labels = to_categorical(encoded_labels, num_classes=num_classes)
        # Divide os dados em treinamento e teste
        return train_test_split(images, labels, test_size=0.2, random_state=42)

    def train_model(self):
        train_gen, val_gen = PreProccessData(
            self.data_dir, target_size=self.input_shape[:2], batch_size=self.batch_size
        ).preprocess_data()

        self.model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        datagen = AugmentData().augment_data()
        datagen.fit(self.x_train)

        history = self.model.fit(train_gen, validation_data=val_gen, epochs=self.epochs)

        self.model.save("tests/saved_models/final_model.keras")
        return history
