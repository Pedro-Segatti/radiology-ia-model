import os
import threading
import json

from kafka import KafkaConsumer, errors

from services.predict_service import PredictService
from kafka_iterators.image_response_producer import ImageResponseProducer

class ImageConsumer:
    def __init__(self):
        kafka_bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9093")
        image_topic = os.getenv("KAFKA_IMAGE_TOPIC", "image-topic")
        response_topic = os.getenv("KAFKA_RESPONSE_TOPIC", "image-response-topic")

        self.consumer = None
        self.producer = ImageResponseProducer()
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.image_topic = image_topic
        self.response_topic = response_topic

        self.initialize_kafka()

    def initialize_kafka(self):
        try:
            self.consumer = KafkaConsumer(
                self.image_topic, bootstrap_servers=self.kafka_bootstrap_servers
            )
            print("Kafka consumer initialized successfully.")
        except errors.NoBrokersAvailable:
            print(f"Could not connect to Kafka brokers at {self.kafka_bootstrap_servers}. Kafka functionality will be disabled.")
        except Exception as e:
            print(f"An unexpected error occurred during Kafka initialization: {e}")

    def start(self):
        if self.consumer:
            thread = threading.Thread(target=self.consume_images)
            thread.start()
        else:
            print("Kafka consumer not initialized. Skipping Kafka consumption.")

    def consume_images(self):
        try:
            if not self.consumer:
                return

            for message in self.consumer:
                try:
                    message_value = json.loads(message.value.decode('utf-8'))
                    conversion_id = message_value['conversion_id']
                    image_data = message_value['image'] 

                    print(f"Received image data: {conversion_id}")

                    _, predicted_class_name, predicted_probability, _ = PredictService.predict_image(image_data)

                    response = {
                        "success": True,
                        "message": "Image Successfully Classified",
                        "data": {
                            "conversion_id": conversion_id,
                            "class": predicted_class_name,
                            "probability": float(predicted_probability)
                        }
                    }

                    self.producer.send_response(json.dumps(response))
                    print(f"Sent response to topic {self.response_topic}: {response}")
                except Exception as e:
                    message_value = json.loads(message.value.decode('utf-8'))
                    conversion_id = message_value['conversion_id']

                    response = {
                        "success": False,
                        "message": "Image Cannot be Classified",
                        "data": {
                            "conversion_id": conversion_id,
                        }
                    }

                    self.producer.send_response(json.dumps(response))
                    print(f"Sent response to topic {self.response_topic}: {response}")

        except Exception as e:
            print(f"An error occurred during Kafka consumption: {e}")