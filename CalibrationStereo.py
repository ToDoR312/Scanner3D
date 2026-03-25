import cv2
import numpy as np
import glob

# PARAMÈTRES DU DAMIER
CHECKERBOARD = (9, 6) # À ajuster selon votre damier
SQUARE_SIZE = 17      # Taille du carré en mm (MESUREZ RÉELLEMENT SUR L'IPAD)

def calibrate_stereo():
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = [] 
    imgpoints_l = [] 
    imgpoints_r = [] 

    images_l = sorted(glob.glob('calib_stereo/left/*.png'))
    images_r = sorted(glob.glob('calib_stereo/right/*.png'))

    for img_l, img_r in zip(images_l, images_r):
        img_l = cv2.imread(img_l)
        img_r = cv2.imread(img_r)
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD)

        if ret_l and ret_r:
            objpoints.append(objp)
            imgpoints_l.append(corners_l)
            imgpoints_r.append(corners_r)
            print(f"Paire valide : {img_l}")

    # On utilise vos matrices intrinsèques déjà calculées
    # (Remplacez par vos valeurs exactes obtenues précédemment)
    mtx_l = np.array([[856.59, 0, 649.69], [0, 857.35, 400.66], [0, 0, 1]])
    dist_l = np.zeros(5) # Mettre vos param_dist ici
    mtx_r = np.array([[859.75, 0, 646.64], [0, 859.33, 376.90], [0, 0, 1]])
    dist_r = np.zeros(5)

    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    ret, m_l, d_l, m_r, d_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r, mtx_l, dist_l, mtx_r, dist_r,
        gray_l.shape[::-1], criteria=criteria, flags=flags)

    print(f"\nERREUR DE REPROJECTION STÉRÉO : {ret}")
    print(f"TRANSLATION (T) en mm : \n{T}")
    
    # Sauvegarde pour la suite (Rectification)
    np.savez("stereo_params.npz", mtx_l=m_l, dist_l=d_l, mtx_r=m_r, dist_r=d_r, R=R, T=T)
    print("\nParamètres sauvegardés dans stereo_params.npz")

if __name__ == "__main__":
    calibrate_stereo()