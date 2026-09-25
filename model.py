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
    p.add_argument('--data-root', default='datasets/all_data')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--img-size', type=int, default=180)
    p.add_argument('--output', default='crop_simple.keras')
    args = p.parse_args()

    data_dir = Path(args.data_root)

    img_size = (args.img_size, args.img_size)

    # load datasets using validation_split
    # We allocate 30% for validation/test, and 70% for training
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, 
        validation_split=0.3,
        subset="training",
        seed=123,
        image_size=img_size, 
        batch_size=args.batch_size
    )
    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, 
        validation_split=0.3,
        subset="validation",
        seed=123,
        image_size=img_size, 
        batch_size=args.batch_size
    )

    # split the 30% val_test_ds into 15% validation and 15% test
    val_batches = tf.data.experimental.cardinality(val_test_ds)
    val_ds = val_test_ds.skip(val_batches // 2)
    test_ds = val_test_ds.take(val_batches // 2)

    num_classes = len(train_ds.class_names)
    print('Classes:', train_ds.class_names)

    model = make_model(num_classes, input_shape=img_size + (3,))
    model.summary()

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    print('Test evaluation:')
    print(model.evaluate(test_ds))

    print("Fine-tuning the model...")
    # Unfreeze the base model inside our main model
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            layer.trainable = True
            
    # Recompile with a very low learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
                  
    model.fit(train_ds, validation_data=val_ds, epochs=10)

    model.save(args.output)
    print('Saved model to', args.output)