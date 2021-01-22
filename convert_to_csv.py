import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import PIL
from PIL import Image
import csv
import ast
import glob

def csvWriter(fil_name, nparray):
    with open(fil_name+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(nparray)

def transformArray(array):
    example = array.tolist()
    final = []
    for i in example:
        final += map(lambda x: x[0], i)
    # remplacer a la main dans le csv 
    final.append(0) # 0 = Femme / 1 = Homme
    return final

if __name__ == "__main__":
    listArrayImages = []
    listImage = glob.glob("./output_images/*.png")
    for img in listImage:
        name = img.split("/")[-1].split(".")[0]
        image = Image.open(img)
        listArrayImages.append(transformArray(np.array(image)))

    # Ecrire image dans csv
    # changer nom du fichier
    csvWriter("myfilename", listArrayImages)
    