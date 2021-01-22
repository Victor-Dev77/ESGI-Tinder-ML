import PIL
from PIL import Image
import glob

if __name__ == "__main__":
    # Changer taille: 28x28 ou 50x50
    size = 28
    # Changer Chemin
    listImage = glob.glob("./input_images/*")
    for img in listImage:
        name = img.split("/")[-1].split(".")[0]
        # Redimensionner image en size x size
        image = Image.open(img).convert('LA')
        wpercent = (size / float(image.size[0]))
        hsize = int((float(image.size[1]) * float(wpercent)))
        image = image.resize((size, size), PIL.Image.ANTIALIAS)
        image.save('./output_images/' + name + '.png')
    