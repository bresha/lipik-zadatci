# Osnovna manipulacijama slikama pomocu Pillow biblioteke

from PIL import Image

print("hello")

# TODO

img = Image.open('vehicle.jpg')

w, h = img.size

img.show()

img_rot = img.rotate(270)
img_rot.show()

w_rot, h_rot = img_rot.size

img_crop = img_rot.crop(((w-h)/2, 0, w_rot-(w-h)/2, h_rot ))

w_crop, h_crop = img_crop.size

img_crop.show()

pasted = Image.new('RGB', (w + w_crop, h_rot), 'white')

pasted.paste(img, (0, 0))
pasted.paste(img_crop, (w, 0))

pasted.show()
pasted.save('vehicle_90.jpg')
