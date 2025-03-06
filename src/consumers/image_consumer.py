import logging
import os
import threading

from kafka import KafkaConsumer, errors

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ImageConsumer:
    def __init__(self):
        kafka_bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        image_topic = os.getenv("KAFKA_IMAGE_TOPIC", "image_topic")

        self.consumer = None
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.image_topic = image_topic

        self.initialize_kafka()

    def initialize_kafka(self):
        try:
            self.consumer = KafkaConsumer(
                self.image_topic, bootstrap_servers=self.kafka_bootstrap_servers
            )
            logging.info("Kafka consumer initialized successfully.")
        except errors.NoBrokersAvailable:
            logging.error(
                f"Could not connect to Kafka brokers at {self.kafka_bootstrap_servers}. Kafka functionality will be disabled."
            )
        except Exception as e:
            logging.error(
                f"An unexpected error occurred during Kafka initialization: {e}"
            )

    def start(self):
        if self.consumer:
            thread = threading.Thread(target=self.consume_images)
            thread.start()
        else:
            logging.warning(
                "Kafka consumer not initialized. Skipping Kafka consumption."
            )

    def consume_images(self):
        try:
            if not self.consumer:
                return

            for message in self.consumer:
                image_data = message.value.decode("utf-8")
                logging.info(f"Received image data: {image_data}")
                # Aqui você chamaria a lógica de predição ou processamento da imagem
        except Exception as e:
            logging.error(f"An error occurred during Kafka consumption: {e}")
