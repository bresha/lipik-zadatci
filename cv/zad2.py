import glob

from PIL import Image, ImageOps

imgPath = "images" 
files = glob.glob(imgPath + '/**/*.jpg', recursive=True)

# TODO ispisite sve putanje
for path in files:
    print(path)


# TODO svaku sliku pretvorite u grayscale i spremite u output direktorij pod nazivom img_x.jpg pri cemu je x redni broj slike
for i, path in enumerate(files):
    img = Image.open(path)
    img_gs = ImageOps.grayscale(img)
    save_path = f'output\\img_{i+1}.jpg'
    img_gs.save(save_path)  