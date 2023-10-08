from tensorflow import keras

# TODO 2: Update the make_model with your model architecture


def make_model(input_shape, num_classes):
    return keras.models.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(num_classes)
    ])



