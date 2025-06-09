from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

class RadiologyModel:
    def __init__(self, input_shape=(256, 256, 1), num_classes=3):
        """
        Inicializa a classe do modelo CNN.
        :param input_shape: Dimensão da imagem de entrada (ex: 256x256x1 para grayscale)
        :param num_classes: Número de classes (ex: compressão, normal, indefinido)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        """
        Constrói a arquitetura da rede neural convolucional.
        """
        model = Sequential()

        # Camadas convolucionais reduzidas
        model.add(Conv2D(32, (3, 3), activation="relu", input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        # Camada densa com regularização e dropout para prevenir overfitting
        model.add(Dense(128, activation="relu", kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.6))

        model.add(Dense(self.num_classes, activation="softmax"))  # 3 saídas

        self.model = model

    def compile_model(
        self, optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    ):
        """
        Compila o modelo com os parâmetros fornecidos.
        """
        if self.model is None:
            raise ValueError("O modelo precisa ser construído antes de ser compilado.")

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def get_model(self):
        """
        Retorna o modelo compilado.
        """
        if self.model is None:
            raise ValueError("O modelo ainda não foi construído.")
        return self.model
