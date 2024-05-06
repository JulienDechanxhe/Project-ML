import cv2
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import torch

"""
    Ce script prend une image (en format jpg) d'un mot et donne en sortie des images (en format jpg) de chaque lettre
    du mot, triées dans l'ordre imposé par le mot. La dernière image correspond au mot en lui-même et peut être oubliée.
    Le principe est le suivant :
    L'image d'entrée est mise en noir et blanc pour bien différencier les contours du fond. On identifie les contours
    et on retire tout ceux qui sont trop fins (bruit). On trie les contours, puis on rogne à chaque fois la figure selon
    la dimension du contour et on enregistre l'image.
"""

def resize_with_padding(image, target_size):
    # Calculer les dimensions du padding
    h, w = image.shape[:2]
    ratio = min(target_size[0] / w, target_size[1] / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    pad_w = (target_size[0] - new_w) // 2
    pad_h = (target_size[1] - new_h) // 2

    # Ajouter le padding
    padded_image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=255)

    # Redimensionner l'image
    resized_image = cv2.resize(padded_image, target_size, interpolation=cv2.INTER_AREA)

    return resized_image

def isolate_letters(filepath, display_contours = True):

    # Charger l'image
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    image = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)

    se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    bg = cv2.morphologyEx(image, cv2.MORPH_DILATE, se)

    out_gray = cv2.divide(image, bg, scale=255)


    out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1]
    out_binary = cv2.bitwise_not(out_binary)

    # Contours detection sur l'image binarisée
    contours, _ = cv2.findContours(out_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrer les contours pour éliminer les plus petits
    contours = [contour for contour in contours if cv2.contourArea(contour) > 50]

    # Trier les contours dans l'ordre du mot
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])


    # Afficher les contours sur l'image
    if display_contours:
        contour_img = cv2.drawContours(cv2.cvtColor(out_binary, cv2.COLOR_GRAY2BGR), contours, -1, (0, 255, 0), 2)
        cv2.imshow("Contours", contour_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Segmenter les lettres
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        # Découper l'image pour obtenir la lettre

        letter = cv2.bitwise_not(out_binary[y:y + h, x:x + w])

        # Compress to 28 x 28 pixelized image
        letter = resize_with_padding(letter, (32, 32))
        
        # Rotate and inverse the image to match the mnist format
        letter = cv2.rotate(cv2.flip(letter,1), cv2.ROTATE_90_COUNTERCLOCKWISE) 
    
        # Dezoom
        letter = cv2.resize(letter, (27, 27))

        letter = cv2.bitwise_not(np.array(letter))

        letter =  np.array(tf.pad(tensor=letter, paddings=[[2, 3], [2, 3]]))
        letter = (letter - np.mean(letter))/np.std(letter)

        letter = np.expand_dims(letter, axis=-1)

        # Ajouter une dimension pour le batch, tranformation en torch.tensor et permutation des index
        letter = torch.tensor(np.expand_dims(letter, axis=0), dtype = torch.float32).permute(0, 3, 1, 2)
        yield letter
