import numpy as np
from astropy.io import fits
import os
    
def DarkFlat_Correction(
        Img_crpt,
        Dark_crpt,
        Flat_crpt,
        output_dir = None
):
    # Cargar y acumular los darks y flats
    if Dark_crpt is not None:
        darks = []
        for archivo in os.listdir(Dark_crpt):
            if archivo.endswith(".fits"):
                with fits.open(os.path.join(Dark_crpt, archivo)) as hdul:
                    dark_data = hdul[0].data.astype(np.uint16)
                    darks.append(dark_data)
        if len(darks) > 0:
            master_dark = np.mean(darks, axis=0)
        else:
            master_dark = 0
    else:
        master_dark = 0

    if Flat_crpt is not None:
        flats = []
        for archivo in os.listdir(Flat_crpt):
            if archivo.endswith(".fits"):
                with fits.open(os.path.join(Flat_crpt, archivo)) as hdul:
                    flat_data = hdul[0].data.astype(np.uint16)
                    flats.append(flat_data)
        if len(flats) > 0:
            master_flat = np.mean(flats, axis=0)
        else:
            master_flat = 0
    else:
        master_flat = 0

    # Comprobar carpetas
    if not os.path.isdir(Img_crpt):
        raise ValueError("Img_crpt debe ser una carpeta")

    os.makedirs(output_dir, exist_ok=True)
    # Procesar imágenes
    for archivo in os.listdir(Img_crpt):
        if archivo.lower().endswith(".fits"):
            img = os.path.join(Img_crpt, archivo)
            with fits.open(img) as hdul:
                img_data = hdul[0].data.astype(np.uint16)
            img_corr = img_data - master_dark - master_flat
            img_corr = np.clip(img_corr, 0, None)
            img_corr = img_corr.astype(np.uint16)
            ruta_salida = os.path.join(output_dir, archivo)
            fits.writeto(ruta_salida, img_corr, overwrite=True)


