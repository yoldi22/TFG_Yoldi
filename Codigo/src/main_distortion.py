import numpy as np
import os 
from Utils.utils import (load_image, UndistImg_Generator,
        DistMap_Generator, ErrorMap_Generator, load_csv,
        Save_Data_Sim, Save_Data_Real, RotMap_Generator, 
        KValuesMap_Generator, Init_Detection_Directories)
from Distortion.DistortionDetector import Distortion_Detector, Displacement_Detector 
from Processing.FlatDark import DarkFlat_Correction
from Processing. OpticCenter import OpticCenter_Detector
import pandas as pd
import matplotlib.pyplot as plt

def Main_real(
        DistImg_crpt,
        MainCarpet,
        Flat_crpt,
        Dark_crpt,
        Processed_Img,
        sen_dim,
        px,
        umbral_dist,
        umbral_distancia, 
        spacing,
        medir_spacing,
        n_ph,
        centro_optico=None
):
    # Indicador de simulación
    simulacion = False
    resultados_data = []

    # Se inicializa la estructura de los directorios
    paths = Init_Detection_Directories(
            MainCarpet=MainCarpet,
            simulacion=simulacion)
    ResultsCsv_path = paths[4]
    
    # Primero, se determina el centro óptico de la lente
    if centro_optico is None:
        centro_optico = OpticCenter_Detector(Flat_crpt)
    
    # Se corrige el dark y se cargan las imagenes
    DarkFlat_Correction(DistImg_crpt, Dark_crpt, Flat_crpt=None, output_dir=Processed_Img)
    DistFits = os.listdir(Processed_Img)

    for i, DistFit in enumerate(DistFits):
        # Se cargan las imagenes
        DistImg_path = os.path.join(Processed_Img,
                       DistFit)
        DistImg      = load_image(DistImg_path)

        # Se detecta la distorsión de la imagen real
        results = Distortion_Detector(
                    img=DistImg,
                    n_ph=n_ph,
                    centro_optico=centro_optico,
                    sen_dim=sen_dim,
                    spacing=spacing,
                    umbral_dist=umbral_dist,
                    simulacion=simulacion,
                    umbral_distancia=umbral_distancia,
                    umbral_min=umbral_min,
                    medir_spacing=medir_spacing)
        k_medida           = results[0] / (px ** 2) #pixels to meters
        errores            = results[1]
        simulacion_erronea = results[2]
        error_type         = simulacion_erronea[1] if simulacion_erronea[0] else None
        centros_sindist    = results[3] #relativos a centro optico, px
        centros_dist       = results[4] #relativos a centro optico, px
        centro_placa       = results[5] #px no relativos a C_o
        components         = results[6] 
        x_i                = results[7] #px^3
        y_i                = results[8] #px
        dist_centro_optimo = results[9] #px distancia del centro fijo al centro optico
        error_spacing      = results[10] #px
        centro_optico      = results[11]
        # Se generan todos los mapas que visualizan los resultados
        UndistImg_img = UndistImg_Generator(
                    img=DistImg,
                    centro_optico=centro_optico,
                    k_medida=k_medida * px ** 2) #meters to pixels
        DistMap_img = DistMap_Generator(
                    fig=DistImg,
                    centros_sindist=centros_sindist,
                    centros_dist=centros_dist,
                    centro_optico=centro_optico,
                    centro_placa=centro_placa,
                    sen_dim=sen_dim)
        RotMap_img = RotMap_Generator(
                    img=DistImg,
                    components=components,
                    centro_placa=centro_placa)
        KValuesMap_img = KValuesMap_Generator(
                    x_i=x_i,
                    y_i=y_i,
                    k_medida=k_medida * px ** 2) #meters to pixels
        Save_Data_Real(
                    DistMap_img=DistMap_img,
                    UndistImg_img=UndistImg_img,
                    KValuesMap_img=KValuesMap_img,
                    RotMap_img=RotMap_img,
                    i=i,
                    paths=paths)

        # Se guardan los datos de los CSV
        resultados_data.append({
            "Number"           : i,
            "Measured_k(m^-2)" : k_medida, #m
            "RMSE(m)"          : errores[2] / (px ** 2), #px to meters 
            "R2"               : errores[3],
            "Error_Type"       : error_type,
            "DistC_o(px)"      : dist_centro_optimo,
            "Error_spacing(px)": error_spacing}) #px

    # Se guardan los datos en el CSV
    df_res = pd.DataFrame(resultados_data)
    df_res.to_csv(ResultsCsv_path, mode='a', header=False, index=False, sep='\t')
    
    # Columnas (base 0) de las que quieres la media
    cols_media_res = [1, 2, 3, 5, 6]

    # Crear fila vacía
    fila_media_res = [''] * df_res.shape[1]

    # Calcular medias solo en las columnas deseadas
    for col in cols_media_res:
        fila_media_res[col] = pd.to_numeric(df_res.iloc[:, col], errors='coerce').mean()

    # Poner etiqueta en la primera columna
    fila_media_res[0] = 'MEDIA'

    # Escribir directamente la fila en los CSV
    pd.DataFrame([fila_media_res]).to_csv(
        ResultsCsv_path,
        mode='a',
        header=False,
        index=False,
        sep='\t')

def Main_sim(
        DistImg_crpt,
        IdealImg_crpt,
        MainCarpet,
        SimCsv_path,
        umbral_dist,
        umbral_distancia,
        umbral_min,
        medir_rot,
        medir_spacing,
        UndistImg_cont,
        DistMap_cont,
        ErrorMap_cont,
        RotMap_cont,
        KValuesMap_cont
):
    simulacion = True

    # Se inicializa la estructura de los directorios
    paths = Init_Detection_Directories(
            MainCarpet=MainCarpet,
            simulacion=simulacion)
    ResultsCsv_path = paths[5]
    ErrorsCsv_path = paths[6]

    # Se lee el csv con todos los datos de la simulación
    df = load_csv(SimCsv_path)
    n_ph_x_arr  = df["n_ph_x"].to_numpy(dtype=int)
    n_ph_y_arr  = df["n_ph_y"].to_numpy(dtype=int)
    co_x_arr    = df["Cntr_optico_x(px)"].to_numpy(dtype=float)
    co_y_arr    = df["Cntr_optico_y(px)"].to_numpy(dtype=float)
    sen_y_arr   = df["SenDim_y"].to_numpy(dtype=int)
    sen_x_arr   = df["SenDim_x"].to_numpy(dtype=int)
    spacing_arr = df["Spacing(m)"].to_numpy(dtype=float)
    k_real_arr  = df["k_real(m^-2)"].to_numpy(dtype=float)
    px_arr      = df["PX(m)"].to_numpy(dtype=float)
    rot_arr     = df["Rot(Rad)"].to_numpy(dtype=float)

    # Se cargan las imagenes de las carpetas
    IdealFits = os.listdir(IdealImg_crpt)
    DistFits = os.listdir(DistImg_crpt)

    # Se inicializan datos
    i = 0
    resultados_data = []
    errores_data = []

    for DistFit, IdealFit in zip(DistFits, IdealFits):

        # Se cargan las imagenes
        DistImg_path  = os.path.join(DistImg_crpt, DistFit)
        IdealImg_path = os.path.join(IdealImg_crpt, IdealFit)
        DistImg       = load_image(DistImg_path)
        IdealImg      = load_image(IdealImg_path)

        # Se leen los datos de la simulacion
        n_ph = [n_ph_x_arr[i], n_ph_y_arr[i]]
        centro_optico = np.array([co_x_arr[i], co_y_arr[i]])
        sen_dim = np.array([sen_y_arr[i], sen_x_arr[i]])
        spacing = spacing_arr[i] #meters
        k_real = k_real_arr[i] #meters
        px = px_arr[i] #meters
        if medir_rot:
            rot = None 
        else:
            rot = rot_arr[i] #rad

        # Se detecta la distorsión de la imagen real
        results = Distortion_Detector(
                    img=DistImg,
                    n_ph=n_ph,
                    sen_dim=sen_dim,
                    centro_optico=centro_optico,
                    spacing=spacing / px, #meters to pixels
                    rot=rot, #rad
                    umbral_dist=umbral_dist,
                    simulacion=simulacion,
                    k_real=k_real * (px ** 2), #meters to pixels
                    umbral_distancia=umbral_distancia,
                    umbral_min=umbral_min,
                    medir_spacing=medir_spacing)
        k_medida           = results[0] / (px ** 2) #pixels to meters
        errores            = results[1]
        simulacion_erronea = results[2]
        error_type         = simulacion_erronea[1] if simulacion_erronea[0] else None
        centros_sindist    = results[3] #relativos a centro optico, px
        centros_dist       = results[4] #relativos a centro optico, px
        centro_placa       = results[5] #px no relativos a C_o
        components         = results[6] 
        x_i                = results[7] #px^3
        y_i                = results[8] #px
        dist_centro_optimo = results[9] #px distancia del centro fijo al centro optico
        error_spacing      = results[10] #px

        # Se generan todos los mapas que visualizan los resultados
        UndistImg_img = UndistImg_Generator(
                    img=DistImg,
                    centro_optico=centro_optico,
                    k_medida=k_medida * px ** 2) #meters to pixels
        centros_ideal, desplazamientos, distancias = Displacement_Detector(
                    img_undist=UndistImg_img,
                    img_ideal=IdealImg,
                    umbral_min=umbral_min,)
        DistMap_img = DistMap_Generator(
                    fig=DistImg,
                    centros_sindist=centros_sindist,
                    centros_dist=centros_dist,
                    centro_optico=centro_optico,
                    centro_placa=centro_placa,
                    sen_dim=sen_dim) 
        ErrorMap_img = ErrorMap_Generator(
                    img_ideal=IdealImg,
                    centros_ideal=centros_ideal,
                    desplazamientos=desplazamientos)        
        RotMap_img = RotMap_Generator(
                    img=DistImg,
                    components=components,
                    centro_placa=centro_placa)        
        KValuesMap_img = KValuesMap_Generator(
                    x_i=x_i,
                    y_i=y_i,
                    k_medida=k_medida * px ** 2, #meters to pixels
                    k_real=k_real * px ** 2) #meters to pixels

        # Se especifica si se guardan o no los mapas e imágenes
        if UndistImg_cont <= i:
            UndistImg_img = None
            plt.close(UndistImg_img)
        if DistMap_cont <= i:
            DistMap_img = None
            plt.close(DistMap_img)
        if ErrorMap_cont <= i:
            ErrorMap_img = None
            plt.close(ErrorMap_img)
        if RotMap_cont <= i:
            RotMap_img = None
            plt.close(RotMap_img)
        if KValuesMap_cont <= i:
            KValuesMap_img = None
            plt.close(KValuesMap_img)
        Save_Data_Sim(
                    DistMap_img=DistMap_img,
                    UndistImg_img=UndistImg_img,
                    ErrorMap_img=ErrorMap_img,
                    KValuesMap_img=KValuesMap_img,
                    RotMap_img=RotMap_img,
                    paths=paths,
                    i=i)

        # Se guardan los datos de los CSV
        resultados_data.append({
            "Number"           : i,
            "Measured_k(m^-2)" : k_medida, #m
            "RMSE(m)"          : errores[2] / (px ** 2) , #px to meters 
            "R2"               : errores[3],
            "Error_Type"       : error_type,
            "DistC_o(px)"      : dist_centro_optimo, #px
            "Error_spacing(px)": error_spacing}) #px
        errores_data.append({
            "Number"           : i,
            "Mn_PX_Error"      : np.mean(distancias),
            "Mx_PX_Error"      : np.max(distancias),
            "R_Error"          : errores[0],
            "MDLD"             : errores[1]})

        i += 1

    # Se guardan los datos en el CSV
    df_res = pd.DataFrame(resultados_data)
    df_res.to_csv(ResultsCsv_path, mode='a', header=False, index=False, sep='\t')
    # Se cargan los resultados al CSV de errores
    df_err = pd.DataFrame(errores_data)
    df_err.to_csv(ErrorsCsv_path, mode="a", header=False, index=False, sep='\t')

    # Columnas (base 0) de las que quieres la media
    cols_media_res = [1, 2, 3, 5, 6]
    cols_media_err = [1, 2, 3, 4]

    # Crear fila vacía
    fila_media_res = [''] * df_res.shape[1]
    fila_media_err = [''] * df_err.shape[1]

    # Calcular medias solo en las columnas deseadas
    for col in cols_media_res:
        fila_media_res[col] = pd.to_numeric(df_res.iloc[:, col], errors='coerce').mean()
    for col in cols_media_err:
        fila_media_err[col] = pd.to_numeric(df_err.iloc[:, col], errors='coerce').mean()

    # Poner etiqueta en la primera columna
    fila_media_res[0] = 'MEDIA'
    fila_media_err[0] = 'MEDIA'

    # Escribir directamente la fila en los CSV
    pd.DataFrame([fila_media_res]).to_csv(
        ResultsCsv_path,
        mode='a',
        header=False,
        index=False,
        sep='\t')
    pd.DataFrame([fila_media_err]).to_csv(
        ErrorsCsv_path,
        mode='a',
        header=False,
        index=False,
        sep='\t')

# RUTAS
DistImg_crpt = "" # Carpeta con las imagenes reales (distorsionadas)
IdealImg_crpt = "" # Carpeta con las imagenes ideales (solo para simulaciones)
MainCarpet = "" # Nombre de la carpeta donde se almacenan todos los resutlados
SimCsv_path = "" # Ruta al archivo CSV con los datos de la simulación
Dark_crpt = "" # Carpeta con los Dark Frames
Flat_crpt = "" # Carpeta con los Flat Frames (si None se )
Processed_Img = "" # Nombre de la carpeta donde se almacenan las imagenes procesadas (despues de pasar el Dark)

# PARÁMETROS
# Parámetros del sensor
px = # Pixelsize en metros 
centro_optico = # Posición centro óptico en px (Si None se calcula con el Flat)
sen_dim_vnir =  # Dimensiones del sesor [dimy, dimx]

# Parámetros placa
spacing_swir = # Espacio entre pinholes en metros
n_ph = # Número de pinholes [ny, nx]

# Parámetros de detección
simulacion =  # Indicador de simulación (True o False)
umbral_dist =  # Umbral Z-Score (umbral clásico: 3)
umbral_distancia =  # Umbral mínimo y máximo desplazamiento pinholes distorsionados
                    # si pinhole disotrsionado mas que max: outlier, si pinhole distorsionado 
                    # menos que min: outlier [min, max]
umbral_min =  # Intensidad mínima para que un pinhole sea detectado

# Parámetros de resultados
# Contador de cuantos mapas/imagenes generar
# Si np.inf se generan para todas las simulaciones
UndistImg_cont = 
DistMap_cont =
ErrorMap_cont = 
RotMap_cont =
KValuesMap_cont =

medir_rot = #Si False se coge el angulo real computado de la simulacion
                #Si True el algoritmo mide el anglo de la placa 
medir_spacing = #Si True se el algoritmo calcula spacing
                    #Si False se coge el spacing proporcionado (no error para sim)

if simulacion:
    Main_sim(
        DistImg_crpt=DistImg_crpt,
        IdealImg_crpt=IdealImg_crpt,
        MainCarpet=MainCarpet,
        SimCsv_path=SimCsv_path,
        umbral_dist=umbral_dist,
        umbral_distancia=umbral_distancia,
        umbral_min=umbral_min,
        medir_rot=medir_rot,
        medir_spacing=medir_spacing,
        UndistImg_cont=UndistImg_cont,
        DistMap_cont=DistMap_cont,
        ErrorMap_cont=ErrorMap_cont,
        RotMap_cont=RotMap_cont,
        KValuesMap_cont=KValuesMap_cont)    
else:
    Main_real(
        DistImg_crpt=DistImg_crpt,
        MainCarpet=MainCarpet,
        Flat_crpt=Flat_crpt,
        Dark_crpt=Dark_crpt,
        Processed_Img=Processed_Img,
        px=px,
        sen_dim=sen_dim,
        umbral_dist=umbral_dist,
        umbral_distancia=umbral_distancia,
        spacing=spacing,
        medir_spacing=medir_spacing,
        n_ph=n_ph,
        centro_optico=centro_optico)


