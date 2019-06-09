import os
from PIL import Image
import piexif

imageFolder = 'training_data/laptops'
listImages = os.listdir(imageFolder)

for img in listImages:
    if img != '.DS_Store':
        imgPath = os.path.join(imageFolder, img)
        # suppose im_path is a valid image path
        #piexif.remove(imgPath)
        try:
            img = Image.open(imgPath)
            exif_data = img.getexif()
        except ValueError as err:
            print(err)
            print("Error on image: ", img)

