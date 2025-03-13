import os

from dotenv import load_dotenv
from flask import Flask, Blueprint
from flask_cors import CORS

from consumers.image_consumer import ImageConsumer
from routes.train import train_bp
from routes.predict import predict_bp

load_dotenv()
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

API_VERSION = os.getenv("API_VERSION", "v1")

api_bp = Blueprint("api", __name__, url_prefix=f"/api/{API_VERSION}")

api_bp.register_blueprint(train_bp, url_prefix="/train")
api_bp.register_blueprint(predict_bp, url_prefix="/predict")

app.register_blueprint(api_bp)

if __name__ == "__main__":
    host = os.getenv("APP_HOST", "localhost")
    port = os.getenv("APP_PORT", "5000")
    app.run(host=host, port=int(port), debug=True)
