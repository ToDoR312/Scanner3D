import torch
import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import matplotlib.pyplot as plt
from typing import Optional, Tuple  # <--- C'est ce qui manquait !

# 1. Chargement des paramètres de TA calibration
try:
    data = np.load("stereo_params.npz")
    K = data['mtx_l']
    dist = data['dist_l']
    T_val = np.linalg.norm(data['T'])
    # Conversion de T en mètres (ex: 80mm -> 0.08m) [cite: 5]
    baseline = T_val / 1000.0 if T_val > 1.0 else T_val
    print(f"[INFO] Calibration chargée. Baseline : {baseline:.3f} m")
except Exception as e:
    print(f"[ERREUR] Paramètres introuvables : {e}")
    exit()

# 2. Préparation de l'IA (Depth Anything V2)
checkpoint = "depth-anything/Depth-Anything-V2-Small-hf"
print(f"[INFO] Chargement de l'IA ({checkpoint})...")
processor = AutoImageProcessor.from_pretrained(checkpoint)
model = AutoModelForDepthEstimation.from_pretrained(checkpoint)

# Utilisation de la carte graphique si dispo
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"[INFO] IA prête sur {device.upper()}")

def split_ps4eye_frame(
    frame: np.ndarray,
    crop_h:   int = 800,
    left_x0:  int = 64,
    right_x0: int = 1280 + 48,
    eye_w:    int = 1264,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Découpe l'image double de la PS4 Eye"""
    if frame is None:
        raise ValueError("Frame vide.")
    h, w = frame.shape[:2]
    use_h = min(h, crop_h)
    fr = frame[:use_h]
    if w >= right_x0 + eye_w:
        left  = fr[:, left_x0  : left_x0  + eye_w].copy()
        right = fr[:, right_x0 : right_x0 + eye_w].copy()
        return left, right
    return fr.copy(), None

def get_metric_cloud(frame):
    """Calcule le nuage de points via l'IA"""
    # a. Correction distorsion
    frame_undistorted = cv2.undistort(frame, K, dist)
    
    # b. Inférence IA
    image = Image.fromarray(cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        depth_map = outputs.predicted_depth.squeeze().cpu().numpy()

    # c. Mise à l'échelle métrique (0.5m à 3m par défaut)
    depth_metric = cv2.normalize(depth_map, None, 0.5, 3.0, cv2.NORM_MINMAX)

    # d. Projection 3D (Pixels -> Mètres)
    f_x, f_y = K[0, 0], K[1, 1]
    c_x, c_y = K[0, 2], K[1, 2]
    rows, cols = depth_metric.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    
    x = (c - c_x) * depth_metric / f_x
    y = (r - c_y) * depth_metric / f_y
    z = depth_metric
    
    return np.stack([x, y, z], axis=-1).reshape(-1, 3)

def show_cloud(points):
    """Affiche le résultat en 3D (échantillonné pour la vitesse)"""
    sample = points[np.random.choice(points.shape[0], 20000, replace=False)]
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sample[:, 0], sample[:, 2], -sample[:, 1], s=1, c=sample[:, 2], cmap='viridis')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (Profondeur m)')
    ax.set_zlabel('Y (m)')
    plt.title("Nuage de points IA")
    plt.show()

# --- BOUCLE PRINCIPALE ---
cap = cv2.VideoCapture(1) # Vérifie l'index (0 ou 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3448)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 808)

print("[INFO] Appuie sur 's' pour scanner, 'q' pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    left, _ = split_ps4eye_frame(frame)
    cv2.imshow("Preview Gauche (Appuyez sur S pour scanner)", cv2.resize(left, (640, 400)))
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        print("[INFO] Scan en cours...")
        points = get_metric_cloud(left)
        show_cloud(points)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()