from cnn.data.pre_proccess_data import PreProccessData


class ValidateTrain:

    def __init__(self, model, data_path, input_shape=(256, 256, 1)):
        self.model = model
        self.data_path = data_path
        self.input_shape = input_shape

    def validate_model(self):
        # Processa os dados e retorna geradores de treinamento e validação, além dos pesos das classes
        _, val_gen = PreProccessData(self.data_path).preprocess_data()

        loss, accuracy = self.model.evaluate(val_gen)

        print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")
        return loss, accuracy

