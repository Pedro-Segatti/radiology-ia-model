import os

import tensorflow as tf
from dotenv import load_dotenv

from cnn.data.predict import Predict
from cnn.models.radiology_model import RadiologyModel
from cnn.train.train import Train
from cnn.train.validate_train import ValidateTrain

load_dotenv()


class TrainService:

    @staticmethod
    def train_model():
        model_path = os.getenv("MODEL_PATH", None)
        if not model_path:
            return

        data_train_path = os.getenv("DATA_TRAIN_PATH", None)
        if not data_train_path:
            return

        radiology_model = None
        try:
            radiology_model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print(e)
            pass

        if not radiology_model:
            radiology_model = RadiologyModel()
            radiology_model.build_model()
            radiology_model.compile_model(
                optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
            )
            radiology_model = radiology_model.get_model()

        train = Train(radiology_model, model_path, data_train_path)
        history = train.train_model()

        validate = ValidateTrain(
            radiology_model,
            data_train_path,
        )
        loss, accuracy = validate.validate_model()

        return history, loss, accuracy


if __name__ == "__main__":
    TrainService.train_model()
