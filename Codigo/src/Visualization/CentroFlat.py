import numpy as np
from astropy.io import fits
from scipy import ndimage
from scipy.optimize import least_squares
import cv2


def _fit_circle_ls(x, y):
    """Ajuste de círculo por mínimos cuadrados"""

    def residuals(p, x, y):
        cx, cy, r = p
        return np.sqrt((x - cx)**2 + (y - cy)**2) - r

    cx0, cy0 = np.mean(x), np.mean(y)
    r0 = np.mean(np.sqrt((x - cx0)**2 + (y - cy0)**2))

    res = least_squares(residuals, [cx0, cy0, r0], args=(x, y))
    return res.x  # cx, cy, r


def OpticalCenter_FromFlat_Circle_Image(
    fits_path,
    iso_level=0.5,
    gaussian_sigma=5,
    thickness_circle=4,
    radius_center=6
):
    """
    Calcula el centro óptico ajustando un círculo a un flat y devuelve
    la imagen con el círculo y el centro dibujados.

    Devuelve
    --------
    centro_optico : (cx, cy)
    img_out : ndarray (uint8, RGB)
    """

    # --- Leer FITS ---
    with fits.open(fits_path) as hdul:
        img = hdul[0].data.astype(np.float64)

    # --- Normalizar ---
    img -= np.min(img)
    img /= np.max(img)

    # --- Suavizado ---
    img_smooth = ndimage.gaussian_filter(img, sigma=gaussian_sigma)

    # --- Extraer contorno iso-intensidad ---
    mask = np.abs(img_smooth - iso_level) < 0.01
    y_pts, x_pts = np.where(mask)

    if len(x_pts) < 100:
        raise RuntimeError("No hay suficientes puntos para ajustar el círculo")

    # --- Ajuste de círculo ---
    cx, cy, r = _fit_circle_ls(x_pts, y_pts)

    # --- Preparar imagen para dibujar ---
    # Aumentar contraste (stretch)
    p2, p98 = np.percentile(img, (2, 98))
    img_stretch = np.clip((img - p2) / (p98 - p2), 0, 1)

    img_8bit = (img_stretch * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2RGB)

    # --- Dibujar círculo ---
    cv2.circle(
        img_rgb,
        (int(cx), int(cy)),
        int(r),
        (255, 0, 0),   # rojo
        thickness_circle
    )

    # --- Dibujar centro óptico: cruz grande ---
    size = 15
    thick = 3

    cv2.line(
        img_rgb,
        (int(cx - size), int(cy)),
        (int(cx + size), int(cy)),
        (0, 255, 0),
        thick
    )
    cv2.line(
        img_rgb,
        (int(cx), int(cy - size)),
        (int(cx), int(cy + size)),
        (0, 255, 0),
        thick
    )

    return (cx, cy), img_rgb

path = r'ImgReales/VNIR90_PSFgrid36/Flats/centro_optico.fits'
centro_optico, img_resultado = OpticalCenter_FromFlat_Circle_Image(
    path
)
cv2.imwrite("flat_centro_optico.png", img_resultado)
