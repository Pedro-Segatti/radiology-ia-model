import traceback
import base64
from io import BytesIO
from flask import Blueprint, Response, jsonify, request
from PIL import Image
from services.predict_service import PredictService

predict_bp = Blueprint("predict", __name__)

@predict_bp.route("/", methods=["POST"])
def train():
    try:
        data = request.get_json()
        image_base64 = data.get("image")

        if not image_base64:
            return "No image provided", 400
        
        _, predicted_class_name, predicted_probability, _ = PredictService.predict_image(image_base64)

        response = {
            "class": predicted_class_name,
            "probability": float(predicted_probability)
        }

        return response
    
    except Exception as e:
        traceback.print_exc()
        raise ValueError(f"Unexpected Error: {e}")
