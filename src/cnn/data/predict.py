import base64
from io import BytesIO

import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img


class Predict:

    def __init__(self, model, target_size=(128, 128, 3)):
        self.model = model
        self.target_size = target_size

    def _load_image_from_path(self, image_path):
        return load_img(image_path, target_size=self.target_size)

    def _load_image_from_base64(self, image_base64):
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data)).convert("RGB")
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

        # Fazer a previsão
        predictions = self.model.predict(image_array)

        # Exibir as classes do modelo
        if hasattr(self.model, "classes_"):
            class_names = self.model.classes_
        else:
            class_names = [f"Classe {i}" for i in range(predictions.shape[1])]
        print(f"Classes disponíveis no modelo: {class_names}")

        # Exibir probabilidades de cada classe
        print("Probabilidades para cada classe:", predictions)

        # Encontrar a classe com maior probabilidade
        predicted_class = np.argmax(predictions)
        predicted_class_name = class_names[predicted_class]
        predicted_probability = predictions[0][predicted_class]

        print(f"Classe predita: {predicted_class_name}")
        print(f"Probabilidade da classe predita: {predicted_probability:.2f}")

        return predicted_class, predicted_class_name, predicted_probability, predictions
