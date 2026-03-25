import cv2
import numpy as np

def split_ps4eye_frame(frame):
    if frame is None: return None, None
    h, w = frame.shape[:2]
    
    # Coordonnées pour extraire les zones utiles
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
        print("Erreur d'ouverture")
        return

    # --- RÉGLAGE DE LA TAILLE D'AFFICHAGE ---
    scale_factor = 0.5

    while True:
        ret, frame = cap.read()
        if not ret: break

        left, right = split_ps4eye_frame(frame)

        if left is not None:
            # Redimensionnement pour l'écran
            disp_l = cv2.resize(left, (0, 0), fx=scale_factor, fy=scale_factor)
            cv2.imshow('Gauche', disp_l)
            
            if right is not None:
                disp_r = cv2.resize(right, (0, 0), fx=scale_factor, fy=scale_factor)
                cv2.imshow('Droite', disp_r)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()