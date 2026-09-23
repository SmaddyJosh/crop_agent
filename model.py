import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models



def make_model(num_classes, input_shape=(180, 180, 3)):
    inputs = layers.Input(shape=input_shape)

    # augmentation 
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)

    # Rescale to [-1, 1] for MobileNetV2
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)

     
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False # freeze base model

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    #  CLI
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='datasets/classify_multiclass')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--img-size', type=int, default=180)
    p.add_argument('--output', default='crop_simple.keras')
    args = p.parse_args()

    train_dir = Path(args.data_root) / 'train'
    valid_dir = Path(args.data_root) / 'valid'
    test_dir = Path(args.data_root) / 'test'

    img_size = (args.img_size, args.img_size)

    # load datasets 
    train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, image_size=img_size, batch_size=args.batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(valid_dir, image_size=img_size, batch_size=args.batch_size)

    num_classes = len(train_ds.class_names)
    print('Classes:', train_ds.class_names)

    model = make_model(num_classes, input_shape=img_size + (3,))
    model.summary()

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    # optional test evaluation
    if test_dir.exists():
        test_ds = tf.keras.utils.image_dataset_from_directory(test_dir, image_size=img_size, batch_size=args.batch_size)
        print('Test evaluation:')
        print(model.evaluate(test_ds))

    model.save(args.output)
    print('Saved model to', args.output)