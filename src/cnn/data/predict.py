import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img


class Predict:

    def __init__(self, model, image_path, target_size=(128, 128, 3)):
        self.model = model
        self.image_path = image_path
        self.target_size = target_size

    def predict(self):
        # Carregar e processar a imagem
        image = load_img(self.image_path, target_size=self.target_size)
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
