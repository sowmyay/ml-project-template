import sys
import datetime

import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split

# TODO 4: Import your dataset, model and transforms
from keras_template.src.models import make_model
from keras_template.src.datasets import DummyDataset, DummyDataGenerator
from keras_template.src.transforms import DummyTransform

train_counter = 0
valid_counter = 0


def main(args):
    if tf.config.list_physical_devices('GPU'):
        print("🐎 Running on GPU(s)", file=sys.stderr)
        config = tf.ConfigProto(device_count={'GPU': args.num_gpus, 'CPU': args.num_workers})
    else:
        print("🐌 Running on CPU(s)", file=sys.stderr)
        config = tf.ConfigProto(device_count={'CPU': args.num_workers})

    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    # TODO 6: Change from DummyTransform to your transforms or inbuilt keras augmentation layers
    transforms = keras.Sequential([DummyTransform(), DummyTransform()])

    # TODO 5: Change from DummyDataset to your dataset or use keras built methods such as image_dataset_from_directory
    train_paths, val_paths = train_test_split(args.dataset, shuffle=False)

    train_dataset = DummyDataset(train_paths, transform=transforms)
    val_dataset = DummyDataset(val_paths, transform=transforms)

    train_generator = DummyDataGenerator(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_generator = DummyDataGenerator(val_dataset, batch_size=args.batch_size, shuffle=False)

    if tf.config.list_physical_devices('GPU'):
        # Prefetching samples in GPU memory helps maximize GPU utilization.
        train_generator = train_generator.prefetch(tf.data.AUTOTUNE)
        val_generator = val_generator.prefetch(tf.data.AUTOTUNE)

    # TODO 7: Change from DummyModel to your model
    model = make_model(input_shape=(28, 28), num_classes=10)

    # TODO 8: Change optimizer and loss functions as needed
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        callbacks.ModelCheckpoint(args.checkpoint, save_best_only=True, monitor='val_loss', mode='min'),
        callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min'),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min'),
        callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]
    model.fit(
        train_generator,
        epochs=args.num_epochs,
        callbacks=callbacks,
        validation_data=val_generator,
    )
