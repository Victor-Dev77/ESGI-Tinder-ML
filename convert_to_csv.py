import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import PIL
from PIL import Image
import cv2
import os
import random
import csv
import ast
import glob

def initCsv(filename):
    print("init csv : " + filename + ".csv")
    with open(filename+".csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

def csvWriter2(filename, nparray, desired):
    lines = nparray.tolist()
    data = []
    for i in lines:
        for j in i:
            data.append(j)
    with open(filename+'.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        data.append(desired)
        print(desired)
        writer.writerow(data)

def resizer(pictureArray, pathFrom, pathDest, desired):
    for i in pictureArray:
        print(f"{i}")
        image = cv2.imread(f"{pathFrom}{i}", 0) #0 = read in gray
        if image is None:
            print(f"Image {i} not read")
            continue
        res = cv2.resize(image, (30, 30), interpolation=cv2.INTER_NEAREST) #INTER_LANCZOS4
        cv2.imwrite(f"{pathDest}{i}", res, [cv2.IMWRITE_JPEG_QUALITY, 100])
        img = np.array(res)
        files.append([img, desired])

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
   #listArrayImages = []
    #listImage = glob.glob("./output_images/*.png")
    #for img in listImage:
     #   name = img.split("/")[-1].split(".")[0]
      #  image = Image.open(img)
       # listArrayImages.append(transformArray(np.array(image)))

    # Ecrire image dans csv
    # changer nom du fichier
    # csvWriter("myfilename", listArrayImages)


    # print("--- NEW METHOD --- ")
    deepfake = True

    csvName, path_input_woman, path_output_woman, path_input_man, path_output_man, = "", "", "", "", ""
    if deepfake:
        csvName = "csv_deepfake"
        path_input_woman = "deepfake_input/woman/"
        path_output_woman = "deepfake_output/woman/"
        path_input_man = "deepfake_input/man/"
        path_output_man = "deepfake_output/man/"
    else:
        csvName = "csv_images"
        path_input_woman = "images/woman/"
        path_output_woman = "resized_images/woman/"
        path_input_man = "images/man/"
        path_output_man = "resized_images/man/"
    initCsv(csvName)
    files = []
    woman = os.listdir(path_input_woman)
    man = os.listdir(path_input_man)

    resizer(woman, path_input_woman, path_output_woman, 0)
    resizer(man, path_input_man, path_output_man, 1)

    random.shuffle(files)
    for file in files:
        csvWriter2(csvName, file[0], file[1])

    
    