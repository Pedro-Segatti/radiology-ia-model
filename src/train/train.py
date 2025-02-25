from data.augment import AugmentData
from data.preproccess import PreProccessData


class Train:
    def __init__(self, model, x_train, x_test, y_train, y_test, data_dir, input_shape, num_classes, epochs=10, batch_size=32):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test        
        self.model = model
        self.data_dir = data_dir
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
    
    def train_model(self):
        train_gen, val_gen = PreProccessData(self.data_dir, target_size=self.input_shape[:2], batch_size=self.batch_size).preprocess_data()
        
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        datagen = AugmentData().augment_data()
        datagen.fit(self.x_train)
        
        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=self.epochs
        )
        
        self.model.save('saved_models/final_model.h5')
        return history
