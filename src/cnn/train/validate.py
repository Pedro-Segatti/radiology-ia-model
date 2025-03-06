from data.preproccess import PreProccessData


class ValidateTrain:

    def __init__(self, model, data_dir, model_path, input_shape):
        self.model = model
        self.data_dir = data_dir
        self.model_path = model_path
        self.input_shape = input_shape

    def validate_model(self):
        _, val_gen = PreProccessData(
            self.data_dir, target_size=self.input_shape[:2]
        ).preprocess_data()

        loss, accuracy = self.model.evaluate(val_gen)
        print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")
        return loss, accuracy
