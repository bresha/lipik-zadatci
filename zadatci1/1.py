
#%%
'''
Stvorite Numpy array objekt iz Python 3D matrice veličine 3x4x2. Tip elemenata Numpy polja mora biti numpy.float64.
Ispišite na ekran:
    • Broj osi
    • Dimenzije polja
    • Ukupni broj elemenata u polju
    • Tip elemenata u polju
'''

import numpy as np

a = np.array([
    [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8]
    ],
    [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8]
    ],
    [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8]
    ]
], dtype=np.float64)

print(a.ndim)
print(a.shape)
print(a.size)
print(a.dtype)

#%%
'''
Promijenite tip Numpy polja iz prethodnog zadatka u np.int16, te ispišite na ekran novi tip polja.
'''

a.dtype = np.int16

print(a.dtype)

#%%
'''
Stvorite 2x7x3 Numpy polje tipa numpy.int32 čiji svi elementi imaju vrijednost 9.
'''

b = np.full((2, 7, 3), 9, dtype=np.int32)

print(b.shape)

#%%
'''
Koristeći odgovarajuću Numpy funkciju, Stvorite Numpy polje 
koje sadrži svaki treći cjelobrojni broj u rasponu [3, 15>.
'''

c = np.arange(3, 15, 3)

print(c)

#%%
'''
Koristeći odgovarajuću Numpy funkciju, Stvorite Numpy polje 
koji sadrži 12 jednoliko raspoređenih brojeva u rasponu [2, 19].
'''

d = np.linspace(2, 19, 12)
print(d)

#%%
'''
Stvorite jediničnu matricu dimenzije 4x4, te još jednu matricu istih 
dimenzija čiji svi elementi imaju vrijednost 9. Ispišite na ekran:
    • Zbroj
    • Razliku
    • Umnožak (element po element)
    • Količnik
'''

e = np.eye(4)
f = np.full((4,4), 9)
print(str(e + f))
print(str(e-f))
print(str(e*f))
print(str(e/f))

#%%
'''
Stvorite 1D ndarray od 10 elemenata proizvoljnih vrijednosti. 
Bez korištenja for petlje postavite svaki drugi element u rasponu indeksa [2, 7> na vrijednost 99.
'''
g = np.ones((11,), dtype=np.int8)
g[2:7:2] = 99

print(g)

#%%
'''
Stvorite 2D ndarray dimenzija 5x6 s elementima proizvoljnih vrijednosti. 
Bez korištelja for petlje izdvojite svaki drugi stupac u zasebnu varijablu. 
Također, u zasebnu varijablu izdvojite posljednja 3 retka.
'''

h = np.array([
    [1, 2, 3, 4, 5, 6],
    [2, 2, 3, 4, 5, 6],
    [3, 2, 3, 4, 5, 6],
    [4, 2, 3, 4, 5, 6],
    [5, 2, 3, 4, 5, 6]
])

i = h[:, 1::2]
print(i)

j = h[-3:]
print(j)

#%%
'''
Stvorite 2D Numpy polje dimenzija 16x16 proizvoljnih vrijednosti. 
Stvorite novo Numpy polje koje sadrži iste elemente prvog polja, ali u obliku 4x8x8. 
Koja Numpy funkcija je prigodna kako bi se ovo ostvarilo? Vizualizirajte oblik novog ndarray-a. 
Dovoljna je skica na papiru ili skica u nekom digitalnom alatu.
'''

k = np.ones((16, 16))
l = k.reshape((4, 8, 8))
print(l.shape)

#%%

'''
Stvorite 3D Numpy polje dimenzija 3x576x720. 
Stvorite novo Numpy polje koje sadrži iste elemente prvog polja, ali u obliku 576x720x3. 
Koja Numpy funkcija je prigodna kako bi se ovo ostvarilo?
'''

m = np.ones((3, 576, 720))
n = m.swapaxes(0, 1)
n = n.swapaxes(1, 2)
print(n.shape)


#%%
'''
Stvorite 2D Numpy polje dimenzija 2x3. 
Ukoliko je srednja vrijednost svih elemenata veća od 10 ispišite sumu svih elemenata, 
u suprotnom ispišite vrijednost najvećeg elementa u polju.
'''
o = np.array([
    [2, 3, 4],
    [5, 6, 7]
])

print(o.sum() if o.mean() > 10 else o.max())
