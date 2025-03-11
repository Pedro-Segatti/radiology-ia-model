import os

import tensorflow as tf
from dotenv import load_dotenv

from cnn.data.predict import Predict

load_dotenv()


class PredictService:

    @staticmethod
    def predict_image(image):
        model_path = os.getenv("MODEL_PATH", None)
        if not model_path:
            return

        model = tf.keras.models.load_model(model_path)
        if not model:
            return

        predict = Predict(model)
        predicted_class, predicted_class_name, predicted_probability, predictions = (
            predict.predict(image, "base64")
        )

        # Validar se é normal ou com problema com base na classe e probabilidade
        if predicted_class_name == "normal" and predicted_probability > 0.5:
            print("A imagem é normal.")
        elif predicted_class_name == "compression" and predicted_probability > 0.5:
            print("A imagem tem problema.")
        else:
            print(
                "O modelo não tem certeza sobre a classificação. Probabilidade abaixo de 50%."
            )

        return ""
