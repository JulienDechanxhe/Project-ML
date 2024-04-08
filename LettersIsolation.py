import cv2
import os
"""
    Ce script prend une image (en format jpg) d'un mot et donne en sortie des images (en format jpg) de chaque lettre
    du mot, triées dans l'ordre imposé par le mot. La dernière image correspond au mot en lui-même et peut être oubliée.
    Le principe est le suivant :
    L'image d'entrée est mise en noir et blanc pour bien différencier les contours du fond. On identifie les contours
    et on retire tout ceux qui sont trop fins (bruit). On trie les contours, puis on rogne à chaque fois la figure selon
    la dimension du contour et on enregistre l'image.
"""
# Loading the picture
PATH = 'C:/Users/Portable/Documents/Master 2/Machine Learning and Big Data Processing/Projet ML/Quantum2.jpg'
Path = 'C:/Users/Portable/Documents/Master 2/Machine Learning and Big Data Processing/Projet ML'
image = cv2.imread(PATH, 0)  # Getting the input image in grayscale

# Setting a value 0 or 1 to each pixel to have an image "black and white" (more easy to then find contours)
_, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# Contours detection : Donne une liste des contours (chaque élément est une liste de points)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Contours filtering to eliminate the smallest ones (criterion : surface area higher than a certain threshold)
contours = [contour for contour in contours if cv2.contourArea(contour) > 500]

# Sorting contours in the order of the words
contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[1], reverse=True)

# Letters segmentation
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)

    # Cropping the image to obtain the letter
    letter = binary[y:y + h, x:x + w]

    # Save each letter
    cv2.imwrite(os.path.join(Path, f'mot_{i}.png.jpg'), letter)
