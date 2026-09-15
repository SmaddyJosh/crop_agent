import argparse
import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


def get_class_counts(train_dir):
    classes = sorted([p.name for p in Path(train_dir).iterdir() if p.is_dir()])
    counts = {c: len(list((Path(train_dir) / c).glob("*"))) for c in classes}
    return classes, counts


def compute_class_weight(counts):
    total = sum(counts.values())
    n_classes = len(counts)
    weights = {}
    for i,(c,n) in enumerate(sorted(counts.items())):
        weights[i] = total / (n_classes * n) if n>0 else 1.0
    return weights


def build_model(input_shape, num_classes):
    data_augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    model = models.Sequential([
        data_augment,
        layers.Rescaling(1./255, input_shape=input_shape),

        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main(args):
    train_dir = Path(args.data_root) / 'train'
    valid_dir = Path(args.data_root) / 'valid'
    test_dir = Path(args.data_root) / 'test'

    classes, counts = get_class_counts(train_dir)
    print('Classes found (train):', classes)
    print('Counts:', counts)

    class_weights = compute_class_weight(counts)
    print('Computed class weights:', class_weights)

    img_size = (args.img_size, args.img_size)
    batch = args.batch_size

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        image_size=img_size,
        batch_size=batch,
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        labels='inferred',
        image_size=img_size,
        batch_size=batch,
        shuffle=False
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    model = build_model(input_shape=img_size + (3,), num_classes=len(classes))
    model.summary()

    cb = [
        callbacks.ModelCheckpoint(args.output, save_best_only=True, monitor='val_accuracy', mode='max'),
        callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=cb
    )

    # evaluate on test set if present
    if test_dir.exists():
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            labels='inferred',
            image_size=img_size,
            batch_size=batch,
            shuffle=False
        ).prefetch(AUTOTUNE)
        loss, acc = model.evaluate(test_ds)
        print(f"Test loss: {loss:.4f}, Test accuracy: {acc:.4f}")

    # save final model
    model.save(args.output)
    print('Saved model to', args.output)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='datasets/classify_multiclass', help='root with train/valid/test subfolders')
    p.add_argument('--img-size', type=int, default=180)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--output', default='crop_multiclass.keras')
    args = p.parse_args()
    main(args)
