import os
from kafka import KafkaProducer, errors
import json

class ImageResponseProducer:
    def __init__(self):
        kafka_bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9093")
        response_topic = os.getenv("KAFKA_RESPONSE_TOPIC", "image-response-topic")

        self.producer = None
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.response_topic = response_topic

        self.initialize_kafka()

    def initialize_kafka(self):
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            print("Kafka producer initialized successfully.")
        except errors.NoBrokersAvailable:
            print(f"Could not connect to Kafka brokers at {self.kafka_bootstrap_servers}. Kafka functionality will be disabled.")
        except Exception as e:
            print(f"An unexpected error occurred during Kafka initialization: {e}")

    def send_response(self, response_data):
        if not self.producer:
            print("Kafka producer not initialized. Skipping message sending.")
            return

        try:
            self.producer.send(self.response_topic, key=self.response_topic.encode('utf-8'), value=response_data)
            print(f"Sent response to topic {self.response_topic}: {response_data}")
        except Exception as e:
            print(f"An error occurred while sending response to Kafka: {e}")

