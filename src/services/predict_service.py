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
            print("1. Model Not Found")
            return

        model = tf.keras.models.load_model(model_path)
        if not model:
            print("2. Model Not Found")
            return

        predict = Predict(model)
        predicted_class, predicted_class_name, predicted_probability, predictions = (
            predict.predict(image, "base64")
        )

        return predicted_class, predicted_class_name, predicted_probability, predictions
