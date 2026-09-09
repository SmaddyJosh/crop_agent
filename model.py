import tensorflow as tf
from tensorflow.keras import layers , models

NUM_CLASSES=3

model=models.Sequential([

    layers.Conv2D(32,(3,3), activation='relu', input_shape=(180,180,3)),
    layers.MaxPooling2D(pool_size=(2,2)),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),

    layers.Conv2D(128,(3,3),activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),


    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')


])

model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#training


model.save('crop_disease.keras')
print('saved')