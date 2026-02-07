import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv
from astropy.io import fits
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150

def Save_Fits(
        img,
        ruta_completa
):
    hdu = fits.PrimaryHDU(img)
    hdu.writeto(ruta_completa, overwrite=True)

def Save_Plots(
        img,
        ruta_completa
):
    img.savefig(ruta_completa, dpi=300, bbox_inches="tight")
    plt.close(img)

def load_csv(
        ruta_csv
):
    """
    Loads the simulation metadata from the given CSV file.

    Args:
        ruta_csv (str): Full path to the 'Simulaciones_data.csv' file.

    Returns:
        pd.DataFrame: DataFrame containing the simulation metadata.
    """
    if not os.path.isfile(ruta_csv):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_csv}")
    
    df = pd.read_csv(ruta_csv, sep='\t')

    return df

def load_image(
        ruta_completa
):
    with fits.open(ruta_completa) as hdul:
        img = hdul[0].data

    return img

def ErrorMap_Generator(
        img_ideal,
        centros_ideal,
        desplazamientos,
        zoom_factor=100,
        dpi=600
):
    x = centros_ideal[:, 0]
    y = centros_ideal[:, 1]
    dx = desplazamientos[:, 0] * zoom_factor
    dy = desplazamientos[:, 1] * zoom_factor

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_ideal, cmap="gray", origin="lower")
    
    q = ax.quiver(x, y, dx, dy, angles="xy", scale_units="xy", scale=1,
                  color="lime", width=0.004)
    
    ax.quiverkey(q, X=0.85, Y=1.05, U=zoom_factor,
                 label=f'{zoom_factor} px ampliado', labelpos='E')

    ax.set_aspect('equal')
    ax.axis("off")
    plt.tight_layout()

    return fig

def DistMap_Generator( 
        fig,
        centros_sindist, #relativos a centro optico
        centros_dist, #relativos a centro optico
        centro_optico,
        centro_placa,
        sen_dim,
        zoom_factor=100  # zoom flechas
):
    # Reescalar los centros 
    centros_sindist = centros_sindist + centro_optico
    centros_dist = centros_dist + centro_optico

    # Calcular desplazamiento en píxeles
    dx = (centros_dist[:, 0] - centros_sindist[:, 0]) * zoom_factor
    dy = (centros_dist[:, 1] - centros_sindist[:, 1]) * zoom_factor
    x = centros_sindist[:, 0] 
    y = centros_sindist[:, 1] 

    # Graficar
    fig_out, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(fig, cmap="gray", origin="lower")
    ax.scatter(centros_sindist[:, 0], centros_sindist[:, 1], 
               s=20, color='cyan', marker='+', linewidths=0.2)
    ax.scatter(centros_dist[:, 0], centros_dist[:, 1], 
               s=20, color='red', marker='+', linewidths=0.2)
    ax.scatter(centro_optico[0], centro_optico[1], 
               s=20, color='green', marker='o')
    ax.scatter(centro_placa[0], centro_placa[1], 
               s=20, color='red', marker='o')
    q = ax.quiver(x, y, dx, dy, angles="xy", scale_units="xy", scale=1,
                  color="cyan", width=0.004, alpha=0.8)

    ax.quiverkey(q, X=0.8, Y=1.05, U=zoom_factor,
                 label=f'{zoom_factor} px ampliado', labelpos='E', color='black')
    
    ax.set_aspect('equal')
    ax.axis("off")
    plt.tight_layout()

    return fig_out
 
def UndistImg_Generator(
        img,
        centro_optico,
        k_medida #px
):
    c_x, c_y = centro_optico
    f = 1  # focal virtual para simulación
    
    cameraMatrix = np.array([
        [f, 0, c_x],
        [0, f, c_y],
        [0, 0, 1]
    ], dtype=np.float32)
    distCoeffs = np.array([k_medida, 0, 0, 0, 0], dtype=np.float32)
    img_undist = cv.undistort(img, cameraMatrix, distCoeffs)

    return img_undist

def KValuesMap_Generator(
        x_i,
        y_i,
        k_medida, #px
        k_real=None #px
):
    x_filt = x_i[1]
    y_filt = y_i[1]
    x = x_i[0]
    y = y_i[0]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    # Dibujar los puntos de la regresión
    #ax.scatter(x, y, s=60, alpha=0.9, edgecolors='k', linewidth=0.2, color = 'red')
    ax.scatter(x_filt, y_filt, s=60, alpha=0.9, edgecolors='k', linewidth=0.2)

    # Dibujar el ajuste de la distorsión y la distorsión real
    lim_x = np.max(x) * 1.1
    ax.set_xlabel('R^3')
    ax.set_ylabel("R'-R")
    ax.plot(
        [0, lim_x],
        [0, lim_x * k_medida],
        color='blue', linewidth=1, label='medida'
    )
    if k_real is not None:
        ax.plot(
            [0, lim_x],
            [0, lim_x * k_real],
            color='red', linewidth=1, label='real'
        )
    plt.tight_layout()
    return plt.gcf() 

def RotMap_Generator(
        img,
        components,
        centro_placa,
        longitud=1500
):

    c_x, c_y = centro_placa
    eje1 = components[0]
    eje2 = components[1]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray", origin="lower")

    # Dibujar eje principal (azul)
    ax.plot(
        [c_x - eje1[0] * longitud, c_x + eje1[0] * longitud],
        [c_y - eje1[1] * longitud, c_y + eje1[1] * longitud],
        color='blue', linewidth=0.25
    )

    # Dibujar eje secundario (verde)
    ax.plot(
        [c_x - eje2[0] * longitud, c_x + eje2[0] * longitud],
        [c_y - eje2[1] * longitud, c_y + eje2[1] * longitud],
        color='green', linewidth=0.25
    )

    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    return fig

def Save_Data_Sim(
        DistMap_img,
        UndistImg_img,
        ErrorMap_img,
        KValuesMap_img,
        RotMap_img,
        paths,
        i
):

    if DistMap_img is not None:
        # Se guardan los paths
        DistMap_crpt = paths[0]
        # Se nombran archivos de los resultados
        DistMap_name = f"img_DistMap_{i:04d}.png"
        # Se cargan las rutas
        DistMap_path = os.path.join(DistMap_crpt, DistMap_name)
        # Se guardan los resultados como imágenes 
        Save_Plots(img=DistMap_img, ruta_completa=DistMap_path)

    if UndistImg_img is not None:
        # Se guardan los paths
        UndistImg_crpt = paths[1]
        # Se nombran archivos de los resultados
        UndistImg_name = f"img_UndistImg_{i:04d}.fits"
        # Se cargan las rutas
        UndistImg_path = os.path.join(UndistImg_crpt, UndistImg_name)
        # Se guardan los resultados como imágenes 
        Save_Fits(img=UndistImg_img, ruta_completa=UndistImg_path)

    if ErrorMap_img is not None:
        # Se guardan los paths
        ErrorMap_crpt = paths[2]
        # Se nombran archivos de los resultados
        ErrorMap_name = f"img_ErrorMap_{i:04d}.png"
        # Se cargan las rutas
        ErrorMap_path = os.path.join(ErrorMap_crpt, ErrorMap_name)
        # Se guardan los resultados como imágenes 
        Save_Plots(img=ErrorMap_img, ruta_completa=ErrorMap_path)

    if KValuesMap_img is not None:
        # Se guardan los paths
        KValuesMap_crpt = paths[3]
        # Se nombran archivos de los resultados
        KValuesMap_name = f"img_KValuesMap_{i:04d}.png"
        # Se cargan las rutas
        KValuesMap_path = os.path.join(KValuesMap_crpt, KValuesMap_name)
        # Se guardan los resultados como imágenes 
        Save_Plots(img=KValuesMap_img, ruta_completa=KValuesMap_path)

    if RotMap_img is not None:
        # Se guardan los paths
        RotMap_crpt = paths[4]
        # Se nombran archivos de los resultados
        RotMap_name = f"img_RotMap_{i:04d}.png"
        # Se cargan las rutas
        RotMap_path = os.path.join(RotMap_crpt, RotMap_name)
        # Se guardan los resultados como imágenes 
        Save_Plots(img=RotMap_img, ruta_completa=RotMap_path)

def Save_Data_Real(
        DistMap_img,
        UndistImg_img,
        KValuesMap_img,
        RotMap_img,
        paths,
        i    
):  
    # Se guardan los paths
    DistMap_crpt = paths[0]
    UndistImg_crpt = paths[1]
    KValuesMap_crpt = paths[2]
    RotMap_crpt = paths[3]

    # Se nombran archivos de los resultados
    UndistImg_name = f"img_UndistImg_{i:04d}.fits"
    DistMap_name = f"img_DistMap_{i:04d}.png"
    KValuesMap_name = f"img_KValuesMap_{i:04d}.png"
    RotMap_name = f"img_RotMap_{i:04d}.png"

    # Se cargan las rutas
    UndistImg_path = os.path.join(UndistImg_crpt, UndistImg_name)
    DistMap_path = os.path.join(DistMap_crpt, DistMap_name)
    KValuesMap_path = os.path.join(KValuesMap_crpt, KValuesMap_name)
    RotMap_path = os.path.join(RotMap_crpt, RotMap_name)

    # Se guardan los resultados como imágenes 
    Save_Fits(img=UndistImg_img, ruta_completa=UndistImg_path)
    Save_Plots(img=DistMap_img, ruta_completa=DistMap_path)
    Save_Plots(img=KValuesMap_img, ruta_completa=KValuesMap_path)
    Save_Plots(img=RotMap_img, ruta_completa=RotMap_path)

def Init_Detection_Directories(
        MainCarpet="Resultados",
        simulacion = False
):
    # Se crean todas las carpetas
    if simulacion:
        Errors_crpt = os.path.join(MainCarpet,
                                    "Errors")
        ErrorMap_crpt = os.path.join(Errors_crpt,
                                    "ErrorMaps")
        os.makedirs(Errors_crpt, exist_ok=True)
        os.makedirs(ErrorMap_crpt, exist_ok=True)
    UndistImg_crpt = os.path.join(MainCarpet,
                                "UndistImgs")
    DistMap_crpt = os.path.join(MainCarpet,
                                "DistMaps")
    RotMap_crpt = os.path.join(MainCarpet,
                                "RotMaps")
    KValuesMap_crpt = os.path.join(MainCarpet,
                                "KValuesMap")
    os.makedirs(UndistImg_crpt, exist_ok=True)
    os.makedirs(DistMap_crpt, exist_ok=True)
    os.makedirs(RotMap_crpt, exist_ok=True)
    os.makedirs(KValuesMap_crpt, exist_ok=True)

    # Se crean los CSV-s
    ResultsCsv_path = os.path.join(MainCarpet,
                                "DataResults.csv")
    df = pd.DataFrame(columns=["Number", "Measured_k(m^-2)",
            "RMSE(m)", "R2", "Error_Type", "DistC_o(px)",
            "Error_spacing(px)"])
    df.to_csv(ResultsCsv_path, index=False, sep='\t')

    if simulacion:
        ErrorsCsv_path = os.path.join(Errors_crpt, 
                                    "Errors.csv")
        df = pd.DataFrame(columns=["Number",
            "Mn_PX_Error", "Mx_PX_Error",
            "R_Error", "MDLD"])
        df.to_csv(ErrorsCsv_path, index=False, sep='\t')

    # Se devuelven todos las rutas
    if simulacion:
        paths = [DistMap_crpt, UndistImg_crpt, 
                ErrorMap_crpt, KValuesMap_crpt,
                RotMap_crpt, ResultsCsv_path,
                ErrorsCsv_path,]
    else:
        paths = [DistMap_crpt,
                UndistImg_crpt,
                KValuesMap_crpt,
                RotMap_crpt,
                ResultsCsv_path]       
    return paths

def Init_Simulation_Directories(
        MainCarpet="Simulaciones"
):
    # Se crean las carpetas
    IdealImg_crpt = os.path.join(MainCarpet, "Imagenes_ideales")
    DistImg_crpt = os.path.join(MainCarpet, "Imagenes_distorsionadas")
    os.makedirs(IdealImg_crpt, exist_ok=True)
    os.makedirs(DistImg_crpt, exist_ok=True)

    # Crear CSV
    DataCsv_path = os.path.join(MainCarpet, "Simulaciones_data.csv")
    df = pd.DataFrame(columns=["Nombre_ideal", 
                    "Nombre_dist", "k_real(m^-2)", 
                    "n_ph_x", "n_ph_y", "Spacing(m)",
                    "Cntr_optico_x(px)","Cntr_optico_y(px)",
                    "Rot(Rad)", "SenDim_x", "SenDim_y", 
                    "PX(m)"])
    df.to_csv(DataCsv_path, index=False, sep='\t')
    
    # Se devuelven los paths
    paths = [IdealImg_crpt, DistImg_crpt, DataCsv_path]

    return paths

def Save_Simulation(
        IdealImg,
        DistImg,
        px,
        i,
        paths
):
    # Se guardan los paths
    IdealImg_crpt = paths[0]
    DistImg_crpt = paths[1]
    # Se nombran archivos de los resultados
    IdealImg_name = f"img_ideal_{i:04d}.fits"
    DistImg_name = f"img_dist_{i:04d}.fits"
    # Se cargan las rutas
    IdealImg_path = os.path.join(IdealImg_crpt, IdealImg_name)
    DistImg_path = os.path.join(DistImg_crpt, DistImg_name)
    # Se guardan los resultados como imágenes
    Save_Fits(IdealImg, IdealImg_path)
    Save_Fits(DistImg, DistImg_path)


