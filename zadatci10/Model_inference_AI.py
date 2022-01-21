'''
Klasifikacija slika koje sadrze rukom pisane brojeve

'''

from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage import color
import matplotlib.image as mpimg
import numpy as np
import cv2
import tensorflow.keras as keras

# ucitavanje slike sa diska
filename = 'test.png'

img = mpimg.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img = resize(img, (28, 28))


# TODO:
# - prikazi ucitanu sliku pomocu matplotlib
# - uvjerite se da slika ima oblika kao u MNIST datasetu -> (bijela znamenka na crnoj pozadini)
plt.figure()
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()


# TODO:
# - transformirajte sliku u vektor odgovarajuce velicine za neuronsku mrezu
img_s = img.astype('float32') / 255
img_s = np.reshape(img_s, (1, 784))

# TODO:
# - ucitajte spremljenu neuronsku mrezu pomocu funkcije keras.models.load_model()
model = keras.models.load_model("FCN")


# TODO:
# - napravi predikciju za ucitanu sliku i ispisi u terminal koju znamenku je prepoznala mreza
net_output = model.predict(img_s)
print(net_output)
label = np.argmax(model.predict(img_s))
label = str(int(label))
print('Na slici je znamenka: ', label)


