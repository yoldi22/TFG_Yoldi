import numpy as np
import cv2 as cv
import random
import pandas as pd
from scipy.ndimage import gaussian_filter, shift, rotate
from astropy.convolution import AiryDisk2DKernel
from scipy import signal
import os
from astropy.io import fits

def Distort(
        img,
        k,
        sen_dim,
        centro_optico
):

    h, w = sen_dim
    f = 1  # focal virtual para simulación
    c_x, c_y = centro_optico

    cameraMatrix = np.array([
        [f, 0, c_x],
        [0, f, c_y],
        [0, 0, 1]
    ], dtype=np.float32)
    
    distCoeffs = np.array([k, 0, 0, 0, 0], dtype=np.float32)

    # 1. Crear cuadrícula regular de puntos (ideal)
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    pts_dist = np.vstack((X.flatten(), Y.flatten())).T.astype(np.float32)

    # 2. Calcular posiciones desdistorsionadas (para remap)
    pts_ud = cv.undistortPoints(pts_dist, cameraMatrix, distCoeffs, None, cameraMatrix)
    pts_ud = pts_ud.reshape(h, w, 2)

    # 3. Crear mapas para remap
    map_x = pts_ud[:, :, 0].astype(np.float32)
    map_y = pts_ud[:, :, 1].astype(np.float32)

    # 4. Aplicar remap → forzar distorsión
    img_dist = cv.remap(img, map_x, map_y, interpolation=cv.INTER_LINEAR)

    return img_dist

def GenerateImage(
        sen_dim,
        n_ph,
        radio,
        delta_w,
        delta_h,
        delta_cx,
        delta_cy,
        delta_degree,
        px,
        centro_optico,
        kpx_max,
        kpx_min,
        I_min,
        I_max        
):
    
    # Se escoge aleatoriamente el numero de ph vertical y
    # horizontal, si no se especifica un número concreto
    if n_ph is None:
        n_x = random.randint(10, 20)
        n_y = random.randint(10, 20)
        n_ph = np.array([n_x, n_y])
    else:
        n_x, n_y = n_ph
                          
    # La dimensión con más pinholes será la horizontal siempre
    if n_x < n_y:
        n_y, n_x = n_x, n_y
        n_ph = np.array([n_x, n_y])

    # Se inicializan la matriz imagen y los parámetros de la imagen
    img = np.zeros(sen_dim)
    dimy, dimx = sen_dim

    # Se calculan los centros de los pinholes ideales
    centros, spacing = Theoretical_Centers(sen_dim=sen_dim, n_ph=n_ph)

    # Cálculo del intervalo de k aceptable
    k_min, k_max = MinMax_k(sen_dim, centros, kpx_max, kpx_min)

    # Se genera k_real (constante de distorsión)
    # en el rango +/-[k_min, k_max]
    if k_min != k_max:
        signo = random.choice([-1, 1])
        k_real = signo * random.uniform(k_min, k_max) 
    else:
        k_real = k_max
        
    # Se dibujan los centros de los pinholes
    for x, y in centros:
        # Se simula una incertidumbre en el brillo
        # de cada pinhole
        if random.random() <= 0.8: # un 20% de pinholes no se dibujan
            img[y,x] = random.randint(I_min, I_max ) 
                                            
    # El centro de la placa se puede detectar
    # añadiendo unos pinholes extra que lo caracterizan
    if n_x % 2 != 0 and n_y % 2 != 0:
        img[dimy // 2,
            dimx // 2 + spacing // 2] = random.randint(I_min, I_max )
        img[dimy // 2,
            dimx // 2 - spacing // 2] = random.randint(I_min, I_max )
        img[dimy // 2 + spacing // 2,
            dimx // 2] = random.randint(I_min, I_max )
        img[dimy // 2 - spacing // 2,
            dimx // 2] = random.randint(I_min, I_max )

    elif n_x % 2 != 0 and n_y % 2 == 0:
        img[dimy // 2,
            dimx // 2] = random.randint(I_min, I_max )
        img[dimy // 2,
            dimx // 2 + spacing // 2] = random.randint(I_min, I_max )
        img[dimy // 2,
            dimx // 2 - spacing // 2] = random.randint(I_min, I_max )

    elif n_x % 2 == 0 and n_y % 2 != 0:
        img[dimy // 2,
            dimx // 2] = random.randint(I_min, I_max )
        img[dimy // 2 + spacing // 2,
            dimx // 2] = random.randint(I_min, I_max )
        img[dimy // 2 - spacing // 2,
            dimx // 2] = random.randint(I_min, I_max )

    else:
        img[dimy // 2, dimx // 2] = random.randint(I_min, I_max)

    # Se simula una incertidumbre de la posición del 
    # centro de la placa respecto al centro de la imagen
    c_x = random.uniform(-delta_w, delta_w) / px 
    c_y = random.uniform(-delta_h, delta_h) / px
    img = shift(img, shift=(c_y, c_x))

    # Se aplica una rotación
    rot = random.uniform(-delta_degree, delta_degree)
    img = rotate(img, rot, reshape=False)

    # Convolución de los pinholes con la PSF (Airy Disk)
    kernel = AiryDisk2DKernel(radio)
    img_convolved = signal.convolve(img, kernel, mode="same")

    # Se añade desenfoque suave
    img_convolved = gaussian_filter(img_convolved, sigma=0.95)

    # Se escala a 12 bits
    img_convolved = np.clip(img_convolved, 0, None)
    img_convolved = (img_convolved / np.max(img_convolved)) * 4095

    # Se simula una incertidumbre de la posición
    # del centro óptico respecto al centro de la imagen
    cntroptc_x = random.uniform(-delta_cx,
                delta_cx) / px + centro_optico[0]
    cntroptc_y = random.uniform(-delta_cy,
                delta_cy) / px + centro_optico[1]
    centro_optico = np.array([cntroptc_x,cntroptc_y])

    #Aplicamos distorsión a la imagen 
    img_dist = Distort(img=img_convolved, k=k_real,
                sen_dim=sen_dim, centro_optico=centro_optico)

    # Añadir ruido gaussiano
    ruido = np.random.normal(loc=0, scale=100, size=sen_dim)
    img_dist += ruido
    img_convolved+=ruido

    # Se escala a 12 bits
    img_dist = np.clip(img_dist, 0, None)
    img_dist = (img_dist / np.max(img_dist)) * 4095
    img_dist = img_dist.astype(np.uint16)

    img_convolved = np.clip(img_convolved, 0, None)
    img_convolved = (img_convolved / np.max(img_convolved)) * 4095
    img_convolved = img_convolved.astype(np.uint16)

    return img_convolved, img_dist, k_real, n_ph, spacing, centro_optico, np.deg2rad(rot)

def Theoretical_Centers(
        sen_dim,
        n_ph,
        margen=0.1
):
    """
    Calcula centros de pinholes con espaciado entero uniforme en píxeles,
    asegurando el mismo espaciado en X e Y y centrado en la imagen.

    Parámetros:
        - sen_dim: [alto_px, ancho_px] del sensor
        - n_ph: [n_x, n_y] numero de pinholes en cada dimensión
        - margen: proporción de margen a dejar en los bordes (opcional)

    Retorna:
        - np.array shape (N, 2) con coordenadas (x_px, y_px) en enteros
    """
    n_x, n_y = n_ph
    alto, ancho = sen_dim
    margen_x = int(margen * ancho)
    margen_y = int(margen * alto)

    # Calculamos el espaciado máximo posible que quepa en ambos ejes
    spacing_x = (ancho - 2 * margen_x) // (n_x - 1)
    spacing_y = (alto - 2 * margen_y) // (n_y - 1)
    spacing = min(spacing_x, spacing_y)

    # Recalculamos margen para centrar la rejilla
    grid_width = spacing * (n_x - 1)
    grid_height = spacing * (n_y - 1)
    start_x = (ancho - grid_width) // 2
    start_y = (alto - grid_height) // 2

    # Generamos coordenadas
    cx = np.arange(n_x) * spacing + start_x
    cy = np.arange(n_y) * spacing + start_y
    X, Y = np.meshgrid(cx, cy)
    centros = np.column_stack((X.ravel(), Y.ravel()))

    return centros, spacing

def MinMax_k(
        sen_dim,
        centros,
        kpx_max,
        kpx_min
):
    h, w = sen_dim
    
    # Primero calculamos el centro del pinhole más esquinado
    # Calcular distancias 
    distancias = np.linalg.norm(centros, axis=1)

    # Obtener el índice del punto más alejado
    indice_max = np.argmax(distancias)

    # Punto más esquinado
    c_x, c_y = centros[indice_max]

    #Calculamos la distancia de este al centro
    r = np.hypot(c_x - w / 2, c_y - h / 2 )

    # K_max será la distorsión correspondiente a un dsplzmnt. de kpx_max pìxeles
    k_max = kpx_max / (r ** 3)

    # k_min será la distorsión correspondiente a un dsplzmnt. de kpx_min píxeles
    k_min = kpx_min / (r ** 3)

    # Devolvemos la k mínima y máxima
    return k_min, k_max
