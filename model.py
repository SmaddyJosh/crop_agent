import argparse
import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path



def make_model(num_classes, input_shape=(180, 180, 3)):
    model = models.Sequential([
        layers.Rescaling(1.0 / 255, input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
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