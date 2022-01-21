import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np

train_ds = image_dataset_from_directory(
    directory='gtsrb_dataset/Train',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(48, 48)
)


test_ds = image_dataset_from_directory(
    directory='gtsrb_dataset/Test_dir',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(48, 48)
)

# for data, labels in train_ds:
#     print(data.shape)



inputs = keras.Input(shape=(48, 48, 3))
x = layers.Rescaling(1./255)(inputs)
x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
x = layers.MaxPool2D()(x)
x = layers.Dropout(0.2)(x)
x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
x = layers.MaxPool2D()(x)
x = layers.Dropout(0.2)(x)
x = layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
x = layers.MaxPool2D()(x)
x = layers.Dropout(0.2)(x)
x = layers.Flatten()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(43, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="gtsrb_cnn")

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_ds, epochs=2)

score = model.evaluate(test_ds)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])


model.save('GTSRB_CNN/')