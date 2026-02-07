import matplotlib.pyplot as plt
from astropy.io import fits
import os
import numpy as np
from Distortion.DistortionDetector import Center_Detector, Detect_Rotation_Spacing, Plate_Center_Detector
from Utils.utils import RotMap_Generator
from Processing.OpticCenter import OpticCenter_Detector
from Processing.FlatDark import DarkFlat_Correction

# Rutas
ruta_imagenes = r"Codigo\DistortionMeasurement\ImgReales\VNIR90_PSFgrid36\Parabola_folder\Zona5\zone05\zone05pos00"
ruta_destino = r"Memoria\TFG_portada-Latex-ehu\Imagenes"
ruta_flat = r"Codigo\DistortionMeasurement\ImgReales\VNIR90_PSFgrid36\Flats"
ruta_dark = r"Codigo\DistortionMeasurement\ImgReales\VNIR90_PSFgrid36\dark"
ruta_procesadas = r"Codigo\DistortionMeasurement\ImgReales\VNIR90_PSFgrid36\Parabola_folder\Zona5\zone05\zone05pos00_processed"

# Parámetros de la simulación
box_dim = 50
ventana = 75
umbral_min = 150
spacing_VNIR90 = 1.1529e-3  #m
spacing = spacing_VNIR90

#DarkFlat_Correction(
#    Img_crpt=ruta_imagenes,
#    Dark_crpt=ruta_dark,
#    Flat_crpt=None,
#    output_dir=ruta_procesadas)

archivo = os.listdir(ruta_procesadas)[0]
if archivo.endswith(".fits"):
    with fits.open(os.path.join(ruta_procesadas, archivo)) as hdul:
        img = hdul[0].data.astype(np.uint16)

optical_center = OpticCenter_Detector(
    ruta_flat)

data = Center_Detector(
    img=img,
    box_dim=box_dim,
    umbral_min=umbral_min,
    ventana=ventana)
candidatos      = data[0]
candidatos_filt = data[1]
subimg          = data[2]
binaria         = data[3]
centroide_local = data[4]
centroides      = data[5]

data = Plate_Center_Detector(centroides)
centro_placa = data[0]
centroides = data[2]

data = Detect_Rotation_Spacing(
    points=centroides,
    optical_center=optical_center,
    spacing=spacing)
angulo     = data[0]
components = data[1]
spacing    = data[2]

fig = RotMap_Generator(
        img=img,
        centro_placa=centro_placa,
        components=components)
fig.savefig(ruta_destino + r"\rotacion.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(angulo, spacing, centroides)