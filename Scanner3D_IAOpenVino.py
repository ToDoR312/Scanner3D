import cv2
import numpy as np
import torch
import openvino as ov
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import matplotlib.pyplot as plt

# --- 1. CHARGEMENT DE TA CALIBRATION ---
try:
    data = np.load("stereo_params.npz")
    K = data['mtx_l']
    dist = data['dist_l']
    print("[INFO] Calibration stéréo chargée.")
except Exception as e:
    print(f"[ERREUR] Fichier calibration introuvable : {e}")
    exit()

# --- 2. PRÉPARATION DU MODÈLE ET CONVERSION OPENVINO ---
checkpoint = "depth-anything/Depth-Anything-V2-Small-hf"
print(f"[INFO] Récupération du modèle {checkpoint}...")

processor = AutoImageProcessor.from_pretrained(checkpoint)
model_torch = AutoModelForDepthEstimation.from_pretrained(checkpoint)
model_torch.eval()

print("[INFO] Conversion vers OpenVINO (Plan B)...")
# Entrée factice pour la conversion (Batch, Canaux, Hauteur, Largeur)
dummy_input = torch.randn(1, 3, 518, 518)
ov_model = ov.convert_model(model_torch, example_input=dummy_input)

# Initialisation du moteur OpenVINO
core = ov.Core()
devices = core.available_devices
print(f"[INFO] Périphériques détectés : {devices}")

# On force l'utilisation du GPU Intel (Iris Xe) si présent, sinon CPU
device_name = "GPU" if "GPU" in devices else "CPU"
compiled_model = core.compile_model(ov_model, device_name)

print(f"[SUCCESS] Modèle chargé sur : {device_name}")

def split_ps4eye_frame(frame):
    """Découpage spécifique PS4 Eye"""
    h, w = frame.shape[:2]
    # Image de gauche pour l'IA (1264x800)
    return frame[:800, 64 : 64 + 1264].copy()

def get_3d_scan_ov(frame):
    """Estimation de profondeur via OpenVINO natif"""
    # Correction optique
    frame_undistorted = cv2.undistort(frame, K, dist)
    rgb = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2RGB)
    
    # Préparation de l'image (518x518)
    input_size = (518, 518)
    img_resized = cv2.resize(rgb, input_size)
    img_input = img_resized.transpose(2, 0, 1) # HWC -> CHW
    img_input = img_input[np.newaxis, ...] / 255.0 
    
    # Inférence
    infer_request = compiled_model.create_infer_request()
    results = infer_request.infer({0: img_input})
    depth_raw = results[0].squeeze()

    # Redimensionnement à la taille de la caméra
    depth_map = cv2.resize(depth_raw, (frame.shape[1], frame.shape[0]))

    # Mise à l'échelle métrique (0.5m à 3.0m)
    depth_metric = cv2.normalize(depth_map, None, 0.5, 3.0, cv2.NORM_MINMAX)

    # --- PROJECTION 3D (SYNTAXE CORRIGÉE) ---
    f_x, f_y = K[0, 0], K[1, 1]
    c_x, c_y = K[0, 2], K[1, 2]
    h, w = depth_metric.shape
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    
    # Calcul des coordonnées XYZ en mètres
    X = (cols - c_x) * depth_metric / f_x
    Y = (rows - c_y) * depth_metric / f_y
    Z = depth_metric
    
    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3) / 255.0
    
    return points, colors

def show_result(pts, cls):
    """Affichage 3D"""
    # Échantillonnage pour la fluidité
    indices = np.random.choice(pts.shape[0], 20000, replace=False)
    p_sample = pts[indices]
    c_sample = cls[indices]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    # Inversion Y pour l'affichage (axe vertical)
    ax.scatter(p_sample[:, 0], p_sample[:, 2], -p_sample[:, 1], s=1, c=c_sample)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (Profondeur m)')
    ax.set_zlabel('Y (m)')
    plt.title("Scan 3D - Méthode Directe OpenVINO")
    plt.show()

# --- BOUCLE PRINCIPALE ---
cap = cv2.VideoCapture(1) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3448)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 808)

print("[INFO] Scanner prêt. 'S' pour scanner, 'Q' pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    left = split_ps4eye_frame(frame)
    cv2.imshow("Apercu - Intel Iris Xe (Plan B)", cv2.resize(left, (640, 400)))
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        print("[INFO] Calcul 3D...")
        pts, cls = get_3d_scan_ov(left)
        show_result(pts, cls)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

