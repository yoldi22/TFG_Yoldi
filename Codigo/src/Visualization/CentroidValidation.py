import matplotlib.pyplot as plt
import os
from astropy.io import fits
import numpy as np
from matplotlib.patches import Rectangle
from Distortion.DistortionDetector import Center_Detector

# Rutas
ruta_imagen = r"Codigo\DistortionMeasurement\ImgReales\girado_90_grados\psf_20000\Zone5_whiteposition_00_processed\IMG_000020000_1.fits"
ruta_destino = r"Memoria\TFG_portada-Latex-ehu\Imagenes"
ruta_imagen = r"Codigo\DistortionMeasurement\ImgReales\VNIR90_PSFgrid36\Parabola_folder\Zona5\zone05\zone05pos00_processed\IMG_000008000_01.fits"
ruta_destino = r"Memoria\TFG_portada-Latex-ehu\Imagenes"
# Parámetros de la simulación
box_dim = 50
ventana = 75
umbral_min = 150

with fits.open(ruta_imagen) as hdul:
    img = hdul[0].data
data = Center_Detector(
        img=img,
        box_dim=box_dim,
        ventana=ventana,
        umbral_min=umbral_min)
candidatos      = data[0]
candidatos_filt = data[1]
subimg          = data[2]
binaria         = data[3]
centroide_local = data[4]
centroides      = data[5]

# Graficar

# Candidatos
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img, cmap="gray", origin="lower")
ax.scatter(
    candidatos[:, 0],
    candidatos[:, 1], 
    s=100,
    color='red',
    marker='+',
    linewidths=0.2)
ax.scatter(
    candidatos_filt[:, 0],
    candidatos_filt[:, 1], 
    s=100,
    color='green',
    marker='+',
    linewidths=0.2)
cx, cy = candidatos_filt[len(candidatos_filt) // 2]  # centro del píxel (enteros)
half = ventana / 2
rect = Rectangle(
    (cx - half - 0.5, cy - half - 0.5),  # esquina inferior izquierda
    ventana,
    ventana,
    linewidth=0.5,
    edgecolor='red',
    facecolor='none')
ax.add_patch(rect)
ax.set_aspect('equal')
ax.axis("off")
plt.tight_layout()
fig.savefig(ruta_destino + r"\Centroide_candidatos_2.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# SUBIMG
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(subimg, cmap="gray", origin="lower")
ax.scatter(
    subimg.shape[1] / 2,
    subimg.shape[0] / 2,
    s=200,
    color='green',
    marker='+')
ax.set_aspect('equal')
ax.axis("off")
plt.tight_layout()
fig.savefig(ruta_destino + r"\Centroide_ventana_2.png", dpi=300, bbox_inches="tight")
plt.close(fig)  

# SUBIMG binaria
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(binaria, cmap="gray", origin="lower")
ax.scatter(
    subimg.shape[1] / 2,
    subimg.shape[0] / 2,
    s=200,
    color='green',
    marker='+')
ax.scatter(
    centroide_local[0],
    centroide_local[1],
    s=200,
    color='blue',
    marker='+')
ax.set_aspect('equal')
ax.axis("off")
plt.tight_layout()
fig.savefig(ruta_destino + r"\Centroide_ventana_binaria_2.png", dpi=300, bbox_inches="tight")
plt.close(fig)  

# Centroides
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img, cmap="gray", origin="lower")
ax.scatter(
    centroides[:, 0],
    centroides[:, 1], 
    s=100,
    color='blue',
    marker='+',
    linewidths=0.2)
ax.set_aspect('equal')
ax.axis("off")
plt.tight_layout()
fig.savefig(ruta_destino + r"\Centroides_2.png", dpi=300, bbox_inches="tight")
plt.close(fig)

