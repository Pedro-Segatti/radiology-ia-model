import json
import traceback

from flask import Blueprint, Response, jsonify

from services.train_service import TrainService
from decorators.default_response import standard_response


train_bp = Blueprint("train", __name__)


@train_bp.route("/", methods=["POST"])
@standard_response
def train():
    try:
        _, loss, accuracy = TrainService.train_model()

        response = {
            "validate_loss": loss,
            "validate_accuracy": accuracy
        }

        return response, 200
    except Exception as e:
        traceback.print_exc()
        raise ValueError(f"Unexpected Error: {e}")

