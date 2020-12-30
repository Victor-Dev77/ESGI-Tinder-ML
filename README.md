# Fonctionnement

1. Executer le fichier img_to_resize_gray.py 

2. Les images dans le dossier output_images sont générées

3. Executer le fichier convert_to_csv.py

4. Le fichier myfilename.csv est généré

5. Dans le fichier tinder-ml.py, changer le nom du fichier a lire: dataset_images.csv en myfilename.csv

6. Executer le fichier tinder-ml.py



# Architecture

## Dosiers:

- image_not_work sont les images qui ne donne pas le meme nb de colonne dans le csv que les autres images donc a la lecture ca crash

- input_images sont les images originales

- output_images sont les images générées en 28x28 ou 50x50 et en noir et blanc


## Fichiers:

- convert_to_csv.py convertit les images du dossiers output_images en 1 fichiers .csv contenant les pixels de tt les images (1 ligne = 1 images).
Le dernier nombre correspond à 0 si Femme, 1 si Homme. A modifier a la main car par defaut ca ajoute 0 pour chaque images.
Ensuite, ajouter d'autres criteres à la suite sur ce qu'on veut detecter dans "desired"

- dataset_images.csv est un exemple de dataset des tp ML

- img_to_resize_gray.py transforme les images du dossiers input_images en une image de taille 28x28 ou 50x50 et en noir et blanc dans le dossier output_images

- tinder-ml.py est le fichier pour faire l'algo de classification

