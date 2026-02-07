from astropy.io import fits
import numpy as np
from scipy.ndimage import center_of_mass
import os
from scipy.optimize import least_squares

def _fit_circle_ls(
    x,
    y
):
    """Ajuste de círculo por mínimos cuadrados"""

    def residuals(
        p,
        x,
        y
    ):
        cx, cy, r = p
        return np.sqrt((x - cx)**2 + (y - cy)**2) - r

    cx0, cy0 = np.mean(x), np.mean(y)
    r0 = np.mean(np.sqrt((x - cx0)**2 + (y - cy0)**2))

    res = least_squares(residuals, [cx0, cy0, r0], args=(x, y))
    return res.x  # cx, cy, r

def OpticCenter_Detector(
        Flat_crpt           
):
    centros_opticos = []
    for archivo in os.listdir(Flat_crpt):
        if archivo.lower().endswith(".fits"):
            img = os.path.join(Flat_crpt, archivo)
            with fits.open(img) as hdul:
                img_data = hdul[0].data.astype(np.uint16)
            # --- Normalizar ---
            img -= np.min(img)
            img /= np.max(img)

            # --- Suavizado ---
            img_smooth = ndimage.gaussian_filter(img, sigma=gaussian_sigma)

            # --- Extraer contorno iso-intensidad ---
            mask = np.abs(img_smooth - iso_level) < 0.01
            y_pts, x_pts = np.where(mask)

            # --- Ajuste de círculo ---
            cx, cy, r = _fit_circle_ls(x_pts, y_pts)

            centros_opticos.append[cx, cy]
            
    return np.mean(centros_opticos, axis=0)
