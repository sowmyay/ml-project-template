from tensorflow.keras.layer import Layer

# TODO 3: Create a your own custom transforms here


class DummyTransform(Layer):
    def __init__(self):
        super().__init__()

    def call(self, images):
        return images
