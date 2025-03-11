import traceback

from flask import Blueprint, Response, jsonify

from services.train_service import TrainService

train_bp = Blueprint("train", __name__)


@train_bp.route("/train", methods=["POST"])
def train():
    try:
        print("TESSSTEEE")
        TrainService.train_model()
        print("TESSSTEEE2")
        return jsonify({"message": "Training hit, data"})
    except Exception:
        traceback.print_exc()
        return Response(
            "Internal Server Error",
            status=500,
        )
