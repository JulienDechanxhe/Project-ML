import cv2
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

"""
    Ce script prend une image (en format jpg) d'un mot et donne en sortie des images (en format jpg) de chaque lettre
    du mot, triées dans l'ordre imposé par le mot. La dernière image correspond au mot en lui-même et peut être oubliée.
    Le principe est le suivant :
    L'image d'entrée est mise en noir et blanc pour bien différencier les contours du fond. On identifie les contours
    et on retire tout ceux qui sont trop fins (bruit). On trie les contours, puis on rogne à chaque fois la figure selon
    la dimension du contour et on enregistre l'image.
"""

def mirror_image(image):
    # Création d'une nouvelle matrice pour l'image miroir
    mirrored_image = [[0 for _ in range(32)] for _ in range(32)]

    # Parcourir l'image d'origine et inverser l'ordre des pixels dans chaque ligne
    for i in range(32):
        # Inverser l'ordre des pixels dans la ligne
        mirrored_image[i] = image[i][::-1]

    return mirrored_image

def rotate_image_90_clockwise(image):
    # Création d'une nouvelle matrice pour l'image pivotée
    rotated_image = [[0 for _ in range(32)] for _ in range(32)]

    # Parcourir l'image d'origine et copier les pixels dans la nouvelle position
    for i in range(32):
        for j in range(32):
            # Calculer la nouvelle position du pixel
            new_i = j
            new_j = 31 - i  # Rotation de 90 degrés dans le sens des aiguilles d'une montre

            # Copier la valeur du pixel dans sa nouvelle position
            rotated_image[new_i][new_j] = image[i][j]

    return rotated_image

def resize_image(image, new_width, new_height):
    # Calculer les facteurs d'échelle en fonction des dimensions d'origine et de destination
    scale_x = new_width / 32
    scale_y = new_height / 32

    # Créer une nouvelle matrice pour l'image redimensionnée
    resized_image = [[0 for _ in range(new_width)] for _ in range(new_height)]

    # Parcourir la nouvelle matrice et interpoler les valeurs de pixel
    for i in range(new_height):
        for j in range(new_width):
            # Calculer les coordonnées correspondantes dans l'image d'origine
            original_x = int(j / scale_x)
            original_y = int(i / scale_y)

            # Copier la valeur du pixel de l'image d'origine à l'emplacement correspondant dans l'image redimensionnée
            resized_image[i][j] = image[original_y][original_x]

    return resized_image

def isolate_letters(filepath):
    # Chemin de l'image
    PATH = os.path.join(filepath)

    # Charger l'image
    image = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)

    image = cv2.GaussianBlur(image, (5, 7), cv2.BORDER_DEFAULT)

    se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    bg = cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
    out_gray = cv2.divide(image, bg, scale=255)
    out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1]
    out_binary = cv2.bitwise_not(out_binary)

    # Afficher l'image binarisée
    # cv2.imshow('Image binarisée', out_binary)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # Contours detection sur l'image binarisée
    contours, _ = cv2.findContours(out_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrer les contours pour éliminer les plus petits
    contours = [contour for contour in contours if cv2.contourArea(contour) > 50]

    # Trier les contours dans l'ordre du mot
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])


    # Afficher les contours sur l'image
    contour_img = cv2.drawContours(cv2.cvtColor(out_binary, cv2.COLOR_GRAY2BGR), contours, -1, (0, 255, 0), 2)
    # cv2.imshow("Contours", contour_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # Segmenter les lettres
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        # Découper l'image pour obtenir la lettre

        letter = out_binary[y:y + h, x:x + w]

        # Compress to 28 x 28 pixelized image
        letter = cv2.resize(letter, (28, 28), interpolation = cv2.INTER_CUBIC)

        letter = (letter - np.mean(letter))/np.std(letter)

        letter =  np.array(tf.pad(tensor=letter, paddings=[[2, 2], [2, 2]]))

        # Rotate and inverse the image to match the mnist format
        letter =  rotate_image_90_clockwise( rotate_image_90_clockwise(rotate_image_90_clockwise( mirror_image(letter))))

        # Dezoom
        letter = resize_image(letter, 27, 27)
        letter =  np.array(tf.pad(tensor=letter, paddings=[[3, 3], [3, 3]]))
    
        yield letter
