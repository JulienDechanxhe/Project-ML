import cv2
import os
import numpy as np
"""
    Ce script prend une image (en format jpg) d'un mot et donne en sortie des images (en format jpg) de chaque lettre
    du mot, triées dans l'ordre imposé par le mot. La dernière image correspond au mot en lui-même et peut être oubliée.
    Le principe est le suivant :
    L'image d'entrée est mise en noir et blanc pour bien différencier les contours du fond. On identifie les contours
    et on retire tout ceux qui sont trop fins (bruit). On trie les contours, puis on rogne à chaque fois la figure selon
    la dimension du contour et on enregistre l'image.
"""
# Loading the picture
PATH = os.path.join(os.getcwd(), "Downloads","HelloWorld.jpg")
Path =  os.path.join(os.getcwd(), "Downloads")

# Charger l'image
image = cv2.imread(PATH, 0)  # Convertir en niveaux de gris

# Filtrer le bruit de fond
background = cv2.GaussianBlur(image, (155, 155), 0)
foreground = cv2.absdiff(image, background)

_, foreground_thresh = cv2.threshold(foreground, 23, 255, cv2.THRESH_BINARY)

# Afficher l'image seuillée après la soustraction du bruit de fond
cv2.imshow('Image seuillée', foreground_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Contours detection
contours, _ = cv2.findContours(foreground_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrer les contours pour éliminer les plus petits
contours = [contour for contour in contours if cv2.contourArea(contour) > 200]

# Trier les contours dans l'ordre du mot
contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

# Afficher les contours sur l'image
contour_img = cv2.drawContours(cv2.cvtColor(foreground_thresh, cv2.COLOR_GRAY2BGR), contours, -1, (0, 255, 0), 2)
cv2.imshow('Contours', contour_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Segmenter les lettres
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)

    # Découper l'image pour obtenir la lettre
    letter = foreground_thresh[y:y + h, x:x + w]

    # Enregistrer chaque lettre
    cv2.imwrite(os.path.join(Path, f'mot_{i}.png.jpg'), letter)