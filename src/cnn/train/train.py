import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from cnn.data.augment import AugmentData


class Train:
    def __init__(
        self,
        model,
        model_path,
        data_path,
        input_shape=(256, 256, 1),
        num_classes=3,
        epochs=75,
        batch_size=2,
        test_size=0.2,
        random_state=42
    ):
        self.model = model
        self.model_path = model_path
        self.data_path = f"{data_path}/train"
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state

        self.x_train, self.x_test, self.y_train, self.y_test = self.load_data()

    def load_data(self):
        """
        Carrega e processa os dados de imagem.

        :return: Tupla contendo os dados de treinamento e teste.
        """
        images = []
        labels = []

        for class_dir in os.listdir(self.data_path):
            class_path = os.path.join(self.data_path, class_dir)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    img = tf.keras.preprocessing.image.load_img(
                        img_path, color_mode='grayscale', target_size=self.input_shape[:2]
                    )
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    images.append(img_array)
                    labels.append(class_dir)

        images = np.array(images, dtype='float32') / 255.0

        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        self.model.classes_ = label_encoder.classes_

        num_classes = len(label_encoder.classes_)
        labels = to_categorical(encoded_labels, num_classes=num_classes)

        return train_test_split(images, labels, test_size=self.test_size, random_state=self.random_state)

    def train_model(self):
        datagen = AugmentData().augment_data()
        datagen.fit(self.x_train)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        history = self.model.fit(
            datagen.flow(self.x_train, self.y_train, batch_size=self.batch_size),
            validation_data=(self.x_test, self.y_test),
            epochs=self.epochs,
            steps_per_epoch=len(self.x_train) // self.batch_size,
            validation_steps=len(self.x_test) // self.batch_size
        )
        self.save_loss_accuracy_plot(history)

        self.model.save(self.model_path)
        return history

    def save_loss_accuracy_plot(self, history, filename="static/loss_accuracy.png"):
        plt.figure(figsize=(10, 5))
        
        # Curva de perda
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Treinamento')
        plt.plot(history.history['val_loss'], label='Validação')
        plt.title("Perda")
        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.legend()

        # Curva de acurácia
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Treinamento')
        plt.plot(history.history['val_accuracy'], label='Validação')
        plt.title("Acurácia")
        plt.xlabel("Época")
        plt.ylabel("Acurácia")
        plt.legend()

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()