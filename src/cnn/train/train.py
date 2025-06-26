import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

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

        self.label_encoder = None
        self.x_train, self.x_test, self.y_train, self.y_test = self.load_data()

    def load_data(self):
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

        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        self.model.classes_ = self.label_encoder.classes_

        labels = to_categorical(encoded_labels, num_classes=len(self.label_encoder.classes_))

        return train_test_split(images, labels, test_size=self.test_size, random_state=self.random_state)

    def train_model(self):
        datagen = AugmentData().augment_data()
        datagen.fit(self.x_train)

        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        checkpoint = ModelCheckpoint(
            self.model_path, monitor='val_loss', save_best_only=True, verbose=1
        )

        # Cálculo dos pesos de classe
        y_integers = np.argmax(self.y_train, axis=1)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_integers),
            y=y_integers
        )
        class_weight_dict = dict(enumerate(class_weights))

        # Compilação do modelo
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        history = self.model.fit(
            datagen.flow(self.x_train, self.y_train, batch_size=self.batch_size),
            validation_data=(self.x_test, self.y_test),
            epochs=self.epochs,
            callbacks=[early_stop, checkpoint],
            class_weight=class_weight_dict
        )

        self.save_loss_accuracy_plot(history)
        return history

    def save_loss_accuracy_plot(self, history, filename="static/loss_accuracy.png"):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Treinamento')
        plt.plot(history.history['val_loss'], label='Validação')
        plt.title("Perda")
        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.legend()

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
