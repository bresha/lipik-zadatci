import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# for i in range(9):
#     plt.subplot(3, 3, i+1)
#     plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
# plt.show()


x_train_s = x_train.astype('float32') / 255
x_test_s = x_test.astype('float32') / 255

y_train_s = keras.utils.to_categorical(y_train, num_classes=10)
y_test_s = keras.utils.to_categorical(y_test, num_classes=10)

model = keras.Sequential()
model.add(layers.Input(shape=(28, 28, 1)))
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train_s, y_train_s, batch_size=32, epochs=5, validation_split=0.2)

plt.figure(1)
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('Epoha')
plt.ylabel('Cross entropy loss')

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.xlabel('Epoha')
plt.ylabel('Accuracy')

plt.legend()
plt.show()

score = model.evaluate(x_test_s, y_test_s)
print('test loss: ', score[0])
print('test accuracy: ', score[1])