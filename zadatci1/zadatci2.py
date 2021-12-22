'''
Pomoću Houghove transformacije detektirajte i nacrtajte pravce na slici “puna_linija_desno.jpg”.
Među detektiranim pravcima mora biti samo desna puna linija.
Izmijenite kôd tako da se među detektiranim linijama također nalaze i isprekidane linije
autoceste.
'''

import cv2
import numpy as np

img = cv2.imread('puna_linija_desno.jpg')