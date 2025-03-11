import os

from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS

from consumers.image_consumer import ImageConsumer
from routes.train import train_bp

load_dotenv()
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# kafka_consumer = ImageConsumer()
# kafka_consumer.start()

app.register_blueprint(train_bp)


if __name__ == "__main__":
    host = os.getenv("APP_HOST", "localhost")
    port = os.getenv("APP_PORT", "5000")
    app.run(host=host, port=int(port), debug=True)
