import sys

import tensorflow as tf
from tensorflow import keras
from keras import layers, callbacks

from tqdm import tqdm

# TODO 9: Import your dataset, model and transforms
from keras_template.src.models import make_model
from keras_template.src.datasets import DummyDataset, DummyDataGenerator
from keras_template.src.transforms import DummyTransform


def main(args):
    if tf.config.list_physical_devices('GPU'):
        print("🐎 Running on GPU(s)", file=sys.stderr)
        config = tf.ConfigProto(device_count={'GPU': args.num_gpus, 'CPU': args.num_workers})
    else:
        print("🐌 Running on CPU(s)", file=sys.stderr)
        config = tf.ConfigProto(device_count={'CPU': args.num_workers})

    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    # TODO 10: Change from DummyModel to your model
    model = make_model(input_shape=(28, 28), num_classes=10)
    model.load_weights(args.checkpoint)

    # TODO 11: Change from DummyTransform to your transforms or inbuilt pytorch transforms
    transforms = keras.Sequential([DummyTransform(), DummyTransform()])

    # TODO 12: Change from DummyDataset to your dataset
    train_dataset = DummyDataset(args.dataset, transform=transforms)
    data_generator = DummyDataGenerator(train_dataset, batch_size=args.batch_size)

    predict(model, data_generator)


def predict(model, data_generator):
    for item in tqdm(data_generator, desc="predict", unit="batch"):
        output = model(item)
    return

