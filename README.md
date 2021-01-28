# Fonctionnement

1. Executer le fichier convert_to_csv.py

2. Changer le boolean "deepfake" a True et re-executer le fichier convert_to_csv.py

3. Les images dans les dossier deepfake_output et resized_images sont générées

4. Les fichiers csv_deepfake.csv et csv_images.csv est générés

5. Executer le fichier tinder-ml.py



# Architecture

## Dosiers:

- deepfake_input sont les images deepfake originales classées entre Homme et Femme

- images sont les images réelles originales classées entre Homme et Femme

- deepfake_output sont les images deepfake générées en 30x30 et en noir et blanc

- resized_images sont les images deepfake générées en 30x30 et en noir et blanc


## Fichiers:

- convert_to_csv.py transforme les images du dossiers deepfake_input et images en image de taille 30x30 et en noir et blanc dans les dossier respectifs deepfake_output et resized_images.

Il convertit aussi les images des dossiers de sorties en fichiers .csv respectifs (csv_deepfake.csv, csv_images.csv) contenant les pixels de toutes les images (1 ligne = 1 images).
Le dernier chiffre de chaque ligne correspond à 0 si l'image est une Femme, 1 si l'image est un Homme.

- csv_deepfake.csv et csv_images.csv sont les fichiers de données à traiter, contenant les images.

- tinder-ml.py est le fichier qui execute l'algorithme de classification

