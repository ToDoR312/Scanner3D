import cv2
import numpy as np
import os

def split_ps4eye_frame(frame):
    if frame is None: return None, None
    h, w = frame.shape[:2]
    
    if w > 2000: # Mode Haute Résolution (3448x808)
        left_x0, right_x0, eye_w, crop_h = 64, 1280 + 48, 1264, 800
    else:        # Mode Basse Résolution (1748x408)
        left_x0, right_x0, eye_w, crop_h = 32, 640 + 24, 632, 400

    use_h = min(h, crop_h)
    left = frame[0:use_h, left_x0 : left_x0 + eye_w]
    right = frame[0:use_h, right_x0 : right_x0 + eye_w] if w >= right_x0 + eye_w else None
    
    return left, right

def main():
    device_index = 1
    cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)

    # Configuration Haute Résolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3448)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 808)

    if not cap.isOpened():
        print("Erreur d'ouverture de la caméra")
        return

    # --- INITIALISATION (Hors de la boucle) ---
    current_side = 'gauche'  # État initial
    scale_factor = 0.7
    
    # Création des dossiers si nécessaire
    for folder in ["calib_gauche", "calib_droite"]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    print(f"Caméra prête. Mode actuel : {current_side}")
    print("Commandes : [S] Sauvegarder | [T] Changer d'objectif | [Q] Quitter")

    while True:
        ret, frame = cap.read()
        if not ret: break

        left, right = split_ps4eye_frame(frame)
        
        # Sélection de l'image à afficher selon l'état actuel
        if current_side == 'gauche':
            active_frame = left
        else:
            active_frame = right

        if active_frame is not None:
            # Affichage
            disp = cv2.resize(active_frame, (0, 0), fx=scale_factor, fy=scale_factor)
            cv2.imshow('Calibration - Vue ' + current_side.capitalize(), disp)

        key = cv2.waitKey(1) & 0xFF
        
        # [S] Sauvegarde
        if key == ord('s'):
            folder = f"calib_{current_side}"
            # Compte le nombre de fichiers déjà présents pour ne pas écraser
            count = len(os.listdir(folder))
            img_name = os.path.join(folder, f"calib_{current_side}_{count:02d}.png")
            
            cv2.imwrite(img_name, active_frame)
            print(f"IMAGE SAUVEGARDÉE : {img_name}")

        # [T] Toggle (Changer de côté)
        elif key == ord('t'):
            current_side = 'droite' if current_side == 'gauche' else 'gauche'
            cv2.destroyAllWindows() # Ferme la fenêtre précédente pour mettre à jour le titre
            print(f"Basculement sur l'objectif : {current_side}")

        # [Q] Quitter
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()