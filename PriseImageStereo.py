import cv2
import os

def split_ps4eye_frame(frame):
    h, w = frame.shape[:2]
    # Découpage spécifique PS4 Eye (Mode 3448x808)
    if w > 2000:
        left_x0, right_x0, eye_w, use_h = 64, 1280 + 48, 1264, 800
    else:
        left_x0, right_x0, eye_w, use_h = 32, 640 + 24, 632, 400
    
    left = frame[0:use_h, left_x0 : left_x0 + eye_w]
    right = frame[0:use_h, right_x0 : right_x0 + eye_w]
    return left, right

def main():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) # Index 1 selon vos précédents messages
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3448)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 808)

    path = "calib_stereo"
    for d in ["left", "right"]: 
        os.makedirs(os.path.join(path, d), exist_ok=True)

    print("--- MODE CAPTURE STÉRÉO ---")
    print("S : Capturer la paire | Q : Quitter")

    while True:
        ret, frame = cap.read()
        if not ret: break

        left, right = split_ps4eye_frame(frame)
        
        # Affichage côte à côte pour contrôle
        combined = cv2.hconcat([cv2.resize(left, (640, 400)), cv2.resize(right, (640, 400))])
        cv2.imshow("Capture Stereo (G | D)", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            count = len(os.listdir(os.path.join(path, "left")))
            cv2.imwrite(os.path.join(path, "left", f"pair_{count:02d}.png"), left)
            cv2.imwrite(os.path.join(path, "right", f"pair_{count:02d}.png"), right)
            print(f"Paire {count} enregistrée.")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()