import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img


class Predict:
    
    def __init__(self, model, image_path, target_size=(64, 64)):
        self.model = model
        self.image_path = image_path
        self.target_size = target_size
    
    def predict(self):
        image = load_img(self.image_path, target_size=self.target_size)
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        prediction = self.model.predict(image_array)
        return np.argmax(prediction), prediction

# if __name__ == '__main__':
#     IMAGE_PATH = 'data/sample_image.jpg'
#     MODEL_PATH = 'saved_models/final_model.h5'
#     predicted_class, confidence = predict(IMAGE_PATH, MODEL_PATH)
#     print(f'Predicted Class: {predicted_class}, Confidence: {confidence}')
