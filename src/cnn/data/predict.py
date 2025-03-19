import base64
from io import BytesIO

import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img


class Predict:

    def __init__(self, model, target_size=(256, 256, 1)):
        self.model = model
        self.target_size = target_size

    def _load_image_from_path(self, image_path):
        return load_img(image_path, color_mode="grayscale", target_size=self.target_size[:2])

    def _load_image_from_base64(self, image_base64):
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data)).convert("L")
        image = image.resize(self.target_size[:2])
        return image

    def predict(self, image_input, input_type="path"):
        if input_type == "path":
            image = self._load_image_from_path(image_input)
        elif input_type == "base64":
            image = self._load_image_from_base64(image_input)
        else:
            raise ValueError("input_type deve ser 'path' ou 'base64'")

        # Processar a imagem
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Adiciona dimensão de canal para imagem em tons de cinza
        if self.target_size[2] == 1:
            image_array = np.expand_dims(image_array, axis=-1)

        # Fazer a previsão
        predictions = self.model.predict(image_array)

        # Exibir as classes do modelo
        if hasattr(self.model, "classes_"):
            class_names = self.model.classes_
        else:
            class_names = ["compression", "normal"]

        # Encontrar a classe com maior probabilidade
        predicted_class = np.argmax(predictions)
        predicted_class_name = class_names[predicted_class]
        predicted_probability = predictions[0][predicted_class]

        return predicted_class, predicted_class_name, predicted_probability, predictions
