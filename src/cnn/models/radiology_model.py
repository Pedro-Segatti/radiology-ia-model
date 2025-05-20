from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential


class RadiologyModel:
    def __init__(self, input_shape=(256, 256, 1), num_classes=3):
        """
        Inicializa a classe do modelo CNN.

        :param input_shape: Tuple representando as dimensões de entrada (altura, largura, canais).
        :param num_classes: Número de classes para a camada de saída.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        """
        Constrói o modelo CNN.
        """
        model = Sequential()

        # Primeira camada convolucional
        model.add(Conv2D(32, (3, 3), activation="relu", input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Segunda camada convolucional
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Terceira camada convolucional
        model.add(Conv2D(128, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Quarta camada convolucional adicional
        model.add(Conv2D(256, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Flatten (camada para transformar em vetor)
        model.add(Flatten())

        # Camada densa totalmente conectada
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.5))  # Dropout para evitar overfitting

        # Camada de saída (usando softmax para classificação)
        model.add(Dense(self.num_classes, activation="softmax"))

        self.model = model

    def compile_model(
        self, optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    ):
        """
        Compila o modelo com os parâmetros fornecidos.

        :param optimizer: Algoritmo de otimização.
        :param loss: Função de perda.
        :param metrics: Lista de métricas para avaliação.
        """
        if self.model is None:
            raise ValueError(
                "O modelo precisa ser construído antes de ser compilado. Use build_model()."
            )

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def get_model(self):
        """
        Retorna o modelo Keras construído.
        """
        if self.model is None:
            raise ValueError("O modelo ainda não foi construído. Use build_model().")

        return self.model
