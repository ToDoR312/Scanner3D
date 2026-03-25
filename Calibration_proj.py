import cv2
import numpy as np
import glob

# -----------------------------
# PARAMETRES
# -----------------------------

projector_size = (1920, 1080)
chessboard_size = (7,6)

# -----------------------------
# CHARGEMENT CALIBRATION CAMERA
# -----------------------------

K_cam = np.load("param_K.npy")
dist_cam = np.load("param_dist.npy")

# -----------------------------
# GENERATEUR GRAYCODE
# -----------------------------
"""
graycode = cv2.structured_light.GrayCodePattern.create(
    projector_size[0],
    projector_size[1]
)

num_patterns = graycode.getNumberOfPatternImages()

print("Nombre de motifs Gray code :", num_patterns)
"""
# -----------------------------
# CHARGEMENT IMAGES CAPTUREES
# -----------------------------

captured_images = []

for i in range(43):

    fname = f"graycode/pattern_{i}.png"

    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise Exception(f"Image manquante : {fname}")

    captured_images.append(img)

# -----------------------------
# DECODAGE CORRESPONDANCES
# -----------------------------

decoded = graycode.decode(captured_images)

if decoded[0] is False:
    raise Exception("Décodage Gray code échoué")

proj_map = decoded[1]

# proj_map contient pour chaque pixel caméra
# la position du pixel projecteur

# -----------------------------
# DETECTION DU DAMIER
# -----------------------------

img = cv2.imread("graycode/reference.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, corners = cv2.findChessboardCorners(gray, chessboard_size)

if not ret:
    raise Exception("Damier non détecté")

corners = cv2.cornerSubPix(
    gray,
    corners,
    (11,11),
    (-1,-1),
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
)

# -----------------------------
# ASSOCIATION CAMERA ↔ PROJECTEUR
# -----------------------------

proj_points = []

for c in corners:

    u = int(c[0][0])
    v = int(c[0][1])

    px = proj_map[v, u, 0]
    py = proj_map[v, u, 1]

    proj_points.append([[px, py]])

proj_points = np.array(proj_points, dtype=np.float32)

# -----------------------------
# POINTS 3D DU DAMIER
# -----------------------------

objp = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)

objpoints = [objp]
proj_imgpoints = [proj_points]

# -----------------------------
# CALIBRATION PROJECTEUR
# -----------------------------

ret, K_proj, dist_proj, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    proj_imgpoints,
    projector_size,
    None,
    None
)

np.save("proj_K.npy", K_proj)
np.save("proj_dist.npy", dist_proj)

print("Calibration projecteur terminée")
print("Matrice projecteur :")
print(K_proj)