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

# Chemin de l'image
PATH = os.path.join(os.getcwd(), "Downloads", "HelloWorld.jpg")
Path = os.path.join(os.getcwd(), "Downloads")

# Charger l'image
image = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)

se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
bg = cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
out_gray = cv2.divide(image, bg, scale=255)
out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1]
out_binary = cv2.bitwise_not(out_binary)
# Afficher l'image binarisée
cv2.imshow('Image binarisée', out_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Remove vertical and horizontal lines

# Contours detection sur l'image binarisée
contours, _ = cv2.findContours(out_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrer les contours pour éliminer les plus petits
contours = [contour for contour in contours if cv2.contourArea(contour) > 150]

# Trier les contours dans l'ordre du mot
contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])


# Afficher les contours sur l'image
contour_img = cv2.drawContours(cv2.cvtColor(out_binary, cv2.COLOR_GRAY2BGR), contours, -1, (0, 255, 0), 2)
cv2.imshow("Contours", contour_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
out_binary= cv2.bitwise_not(out_binary)
# Segmenter les lettres
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    # Découper l'image pour obtenir la lettre

    letter = out_binary[y:y + h, x:x + w]

    # Enregistrer chaque lettre
    cv2.imwrite(os.path.join(Path, f'mot_{i}.png.jpg'), letter)

def jpg_to_mnist(filepath):
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(image, (28, 28), interpolation = cv2.INTER_CUBIC)

# for i in range(11):
mnist_W = jpg_to_mnist(os.path.join(Path, f'mot_{0}.png.jpg'))
import matplotlib.pyplot as plt
plt.imshow(mnist_W, cmap="gray")
plt.show()
