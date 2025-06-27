import base64
from io import BytesIO
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img


class Predict:
    def __init__(self, model, target_size=(256, 256, 1), confidence_threshold=0.6, colorfulness_threshold=0.3):
        self.model = model
        self.target_size = target_size
        self.confidence_threshold = confidence_threshold
        self.colorfulness_threshold = colorfulness_threshold

    def _is_colorful(self, image_pil):
        arr = np.asarray(image_pil.convert("RGB"), dtype=np.float32) / 255.0
        std_channels = np.std(arr, axis=(0, 1))
        colorfulness_score = np.mean(std_channels)
        return colorfulness_score > self.colorfulness_threshold

    def _load_image_from_path(self, image_path, raw=False):
        img = Image.open(image_path)
        if raw:
            return img
        return img.convert("L").resize(self.target_size[:2])

    def _decode_base64_to_pil(self, image_base64):
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        image_data = base64.b64decode(image_base64)
        return Image.open(BytesIO(image_data))

    def _load_image_from_base64(self, image_base64):
        return self._decode_base64_to_pil(image_base64).convert("L").resize(self.target_size[:2])

    def predict(self, image_input, input_type="path"):
        if input_type == "path":
            pil_raw = self._load_image_from_path(image_input, raw=True)
        elif input_type == "base64":
            pil_raw = self._decode_base64_to_pil(image_input)
        else:
            raise ValueError("input_type deve ser 'path' ou 'base64'")

        if self._is_colorful(pil_raw):
            return None, "undefined", 1.0, None

        image = pil_raw.convert("L").resize(self.target_size[:2])
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        if self.target_size[2] == 1 and image_array.shape[-1] != 1:
            image_array = np.expand_dims(image_array, axis=-1)

        # Fazer predição
        predictions = self.model.predict(image_array)
        predicted_class = np.argmax(predictions)
        predicted_probability = predictions[0][predicted_class]

        # Nome das classes
        if hasattr(self.model, "classes_"):
            class_names = self.model.classes_
        else:
            class_names = ["compression", "normal", "undefined"]

        # Classificação com limiar
        if predicted_probability < self.confidence_threshold:
            predicted_class_name = "undefined"
        else:
            predicted_class_name = class_names[predicted_class]

        return predicted_class, predicted_class_name, predicted_probability, predictions
