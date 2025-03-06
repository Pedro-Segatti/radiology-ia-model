from flask import Blueprint, jsonify, request

train_bp = Blueprint("train", __name__)


@train_bp.route("/train", methods=["POST"])
def train():
    data = request.json
    # Lógica de treinamento seria chamada aqui
    return jsonify({"message": "Training endpoint hit, data"})
