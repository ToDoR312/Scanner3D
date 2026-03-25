import cv2
import numpy as np
import os

# -----------------------------
# PARAMETRES PROJECTEUR
# -----------------------------

proj_width = 1280
proj_height = 800

output_dir = "patterns"

os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# CREATION GENERATEUR GRAY CODE
# -----------------------------

gray = cv2.structured_light.GrayCodePattern.create(
    proj_width,
    proj_height
)

# nombre total de motifs
num_patterns = gray.getNumberOfPatternImages()

print("Nombre total de motifs :", num_patterns)

# -----------------------------
# GENERATION DES MOTIFS
# -----------------------------

patterns = gray.generate()

for i, pattern in enumerate(patterns):

    filename = os.path.join(output_dir, f"pattern_{i:03d}.png")

    cv2.imwrite(filename, pattern)

print("Motifs générés dans :", output_dir)

# -----------------------------
# IMAGES NOIR ET BLANC
# (utiles pour le décodage)
# -----------------------------

white = np.full((proj_height, proj_width), 255, dtype=np.uint8)
black = np.zeros((proj_height, proj_width), dtype=np.uint8)

cv2.imwrite(os.path.join(output_dir, "white.png"), white)
cv2.imwrite(os.path.join(output_dir, "black.png"), black)

print("Images white / black ajoutées")