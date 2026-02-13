import numpy as np
import os 
from Utils.utils import (load_image, UndistImg_Generator,
        DistMap_Generator, ErrorMap_Generator, load_csv,
        Save_Data_Sim, Save_Data_Real, RotMap_Generator, 
        KValuesMap_Generator, Init_Detection_Directories)
from Distortion.DistortionDetector import Distortion_Detector, Displacement_Detector 
from Processing.FlatDark import DarkFlat_Correction
from Processing. OpticCenter import OpticCenter_Detector
import time
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
    if Flat_crpt is not None:
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

    total_start = time.perf_counter()  # tiempo total del bucle

    for DistFit, IdealFit in zip(DistFits, IdealFits):

        img_start = time.perf_counter()  # tiempo por imagen

        t0 = time.perf_counter()
        # Se cargan las imagenes
        DistImg_path  = os.path.join(DistImg_crpt, DistFit)
        IdealImg_path = os.path.join(IdealImg_crpt, IdealFit)
        DistImg       = load_image(DistImg_path)
        IdealImg      = load_image(IdealImg_path)
        t_img_load = time.perf_counter() - t0

        # Se leen los datos de la simulacion
        t0 = time.perf_counter()
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
        t_csv_read = time.perf_counter() - t0

        # Se detecta la distorsión de la imagen real
        t0 = time.perf_counter()
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
        t_distortion = time.perf_counter() - t0

        t0 = time.perf_counter()
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
        t_maps = time.perf_counter() - t0

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
        t0 = time.perf_counter()
        Save_Data_Sim(
                    DistMap_img=DistMap_img,
                    UndistImg_img=UndistImg_img,
                    ErrorMap_img=ErrorMap_img,
                    KValuesMap_img=KValuesMap_img,
                    RotMap_img=RotMap_img,
                    paths=paths,
                    i=i)
        t_save = time.perf_counter() - t0

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
        t_img_total = time.perf_counter() - img_start
        print(f"[Image {i}] load: {t_img_load:.2f}s, CSV: {t_csv_read:.2f}s, "
              f"distortion: {t_distortion:.2f}s, maps: {t_maps:.2f}s, save: {t_save:.2f}s, total: {t_img_total:.2f}s")

        i += 1

    total_time = time.perf_counter() - total_start
    print(f"Total processing time: {total_time:.2f}s for {i} images")

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

# PATHS

# Paths VNIR170
DistImg_crpt = r"Data\Reales\VNIR170_PSFgrid\Test1_parab\Parabola_folder\Zona5\zone05\zone05pos00"
IdealImg_crpt = ""
MainCarpet = r"Results\Reales\VNIR170\Results_zone05pos00_prueba"
SimCsv_path = "" 
Dark_crpt = r"Data\Reales\VNIR170_PSFgrid\Test1_parab\dark"
Flat_crpt = r"Data\Reales\VNIR170_PSFgrid\Flats"
Processed_Img = r"Data\Reales\VNIR170_PSFgrid\Test1_parab\Parabola_folder\Zona5\zone05\zone05pos00_processed"

# Paths MIKEL y DAVID esquina 2
DistImg_crpt = r"Esquina 2\Zone5_whiteposition_00"
IdealImg_crpt = ""
MainCarpet = r"Results\Reales\Esquina 2\Results_Zone5_whiteposition_00"
SimCsv_path = "" 
Dark_crpt = r"Esquina1\dark"
Flat_crpt = None
Processed_Img = r"Esquina 2\Zone5_whiteposition_00_processed"

# Paths MDLD
DistImg_crpt = r"Simulaciones_tfg\prueba_MDLD\Imagenes_distorsionadas"
IdealImg_crpt = r"Simulaciones_tfg\prueba_MDLD\Imagenes_ideales"
MainCarpet = r"Simulaciones_tfg\prueba_MDLD"
SimCsv_path = r"Simulaciones_tfg\prueba_MDLD\Simulaciones_data.csv" 
Dark_crpt = ""
Flat_crpt = None
Processed_Img = ""

# Paths MDLD
DistImg_crpt = r"Simulaciones_tfg\prueba_pocos\Imagenes_distorsionadas"
IdealImg_crpt = r"Simulaciones_tfg\prueba_pocos\Imagenes_ideales"
MainCarpet = r"Simulaciones_tfg\prueba_pocos"
SimCsv_path = r"Simulaciones_tfg\prueba_pocos\Simulaciones_data.csv" 
Dark_crpt = ""
Flat_crpt = None
Processed_Img = ""

# Paths ROT 0 DSPLZMNT 0
DistImg_crpt = r"Simulaciones_tfg\ROT_0_Spacing_perfecto_Dsplzmt_0\Imagenes_distorsionadas"
IdealImg_crpt = r"Simulaciones_tfg\ROT_0_Spacing_perfecto_Dsplzmt_0\Imagenes_ideales"
MainCarpet = r"Simulaciones_tfg\ROT_0_Spacing_perfecto_Dsplzmt_0"
SimCsv_path = r"Simulaciones_tfg\ROT_0_Spacing_perfecto_Dsplzmt_0\Simulaciones_data.csv" 
Dark_crpt = ""
Flat_crpt = None
Processed_Img = ""

# Paths ROT 0 
DistImg_crpt = r"Simulaciones_tfg\ROT_0\Imagenes_distorsionadas"
IdealImg_crpt = r"Simulaciones_tfg\ROT_0\Imagenes_ideales"
MainCarpet = r"Simulaciones_tfg\ROT_0"
SimCsv_path = r"Simulaciones_tfg\ROT_0\Simulaciones_data.csv" 
Dark_crpt = ""
Flat_crpt = None
Processed_Img = ""

# Paths NADAFIJO_SPACING_MEDIDO
DistImg_crpt = r"Simulaciones_tfg\NADAFIJO\Imagenes_distorsionadas"
IdealImg_crpt = r"Simulaciones_tfg\NADAFIJO\Imagenes_ideales"
MainCarpet = r"Simulaciones_tfg\NADAFIJO"
SimCsv_path = r"Simulaciones_tfg\NADAFIJO\Simulaciones_data.csv" 
Dark_crpt = ""
Flat_crpt = None
Processed_Img = ""

# Paths TODOFIJO
DistImg_crpt = r"Simulaciones_tfg\NADAFIJO_spacing\Imagenes_distorsionadas"
IdealImg_crpt = r"Simulaciones_tfg\NADAFIJO_spacing\Imagenes_ideales"
MainCarpet = r"Simulaciones_tfg\NADAFIJO_spacing"
SimCsv_path = r"Simulaciones_tfg\NADAFIJO_spacing\Simulaciones_data.csv" 
Dark_crpt = ""
Flat_crpt = None
Processed_Img = ""

# Paths VNIR90
DistImg_crpt = r"ImgReales\VNIR90_PSFgrid36\Parabola_folder\Zona5\zone05\zone05pos00"
IdealImg_crpt = ""
MainCarpet = r"ResultsReales\VNIR90\Results_zone05pos00"
SimCsv_path = "" 
Dark_crpt = r"ImgReales\VNIR90_PSFgrid36\dark"
Flat_crpt = r"ImgReales\VNIR90_PSFgrid36\Flats"
Processed_Img = r"ImgReales\VNIR90_PSFgrid36\Parabola_folder\Zona5\zone05\zone05pos00_processed"

# Paths MIKEL y DAVID esquina 1
DistImg_crpt = r"ImgReales\Esquina1\Zone5_whiteposition_00"
IdealImg_crpt = ""
MainCarpet = r"ResultsReales\Esquina1\Results_Zone5_whiteposition_00"
SimCsv_path = "" 
Dark_crpt = r"ImgReales\Esquina1\dark"
Flat_crpt = None
Processed_Img = r"ImgReales\Esquina1\Zone5_whiteposition_00_processed"

# Paths SWIR90
DistImg_crpt = r"ImgReales\SWIR90_PSFgrid21\Test2_tanda50imagenes_enfocadas\white_Newton_temp_22.0position_00"
IdealImg_crpt = ""
MainCarpet = r"ResultsReales\SWIR90_3\white_Newton_temp_22.1position_00"
SimCsv_path = "" 
Dark_crpt = r"ImgReales\SWIR90_PSFgrid21\Test2_tanda50imagenes_enfocadas\dark"
Flat_crpt = None
Processed_Img = r"ImgReales\SWIR90_PSFgrid21\Test2_tanda50imagenes_enfocadas\white_Newton_temp_22.0position_00_proccesed"

# Paths NADAFIJO_SPACING_REAL
DistImg_crpt = r"c:\Users\xabie\Documents\TFG\Codigo\DistortionMeasurement\Simulaciones_tfg\NADAFIJO_intensidad_2\Imagenes_distorsionadas"
IdealImg_crpt = r"c:\Users\xabie\Documents\TFG\Codigo\DistortionMeasurement\Simulaciones_tfg\NADAFIJO_intensidad_2_2\Imagenes_ideales"
MainCarpet = r"Simulaciones_tfg\NADAFIJO_intensidad_2"
SimCsv_path = r"c:\Users\xabie\Documents\TFG\Codigo\DistortionMeasurement\Simulaciones_tfg\NADAFIJO_intensidad_2\Simulaciones_data.csv" 
Dark_crpt = ""
Flat_crpt = None
Processed_Img = ""

# Paths prueba
DistImg_crpt = r"DatosResultados\Simulaciones_tfg\prueba6\Imagenes_distorsionadas"
IdealImg_crpt = r"DatosResultados\Simulaciones_tfg\prueba6\Imagenes_ideales"
MainCarpet = r"DatosResultados\Simulaciones_tfg\prueba8.2"
SimCsv_path = r"DatosResultados\Simulaciones_tfg\prueba6\Simulaciones_data.csv" 
Dark_crpt = ""
Flat_crpt = None
Processed_Img = ""

# Paths girado90 2000
DistImg_crpt = r"DatosResultados\ImgReales\girado_90_grados\psf_20000\Zone5_whiteposition_00"
IdealImg_crpt = ""
MainCarpet = r"DatosResultados\ResultsReales\girado_90_grados_3\psf_20000\Zone5_whiteposition_00"
SimCsv_path = "" 
Dark_crpt = r"DatosResultados\ImgReales\girado_90_grados\dark_20000\Zone5_whiteposition_00"
Flat_crpt = None
Processed_Img = r"DatosResultados\ImgReales\girado_90_grados\psf_20000\Zone5_whiteposition_00_processed"

# PARAMETERS
# Sensor parameters
px_sim = 1e-6 #m/px # Pixelsize in meters
px_VNIR = 5.5e-6 #m/px 
px = px_VNIR
centro_optico= None  # px # Optic center of the system
sen_dim_vnir = [3072, 4096] # Sensor dimension
sen_dim_swir = [1280, 1024]
sen_dim = sen_dim_vnir

# Plate parameters
spacing_swir = 3.8024662e-4 #m # Sapcing between pinholes
spacing_VNIR170 = 1.1495e-3 #m
spacing_VNIR90 = 1.1529e-3  #m
spacing_MD = 1.175e-3 #m
spacing_girado90 = 1.1371e-3
n_ph = [20, 15] # Number of pinholes in each dimension
spacing = spacing_girado90 / px #meters to px

# Detection parameters
simulacion = False # Simulation index
umbral_dist = 3 # Z-Score threshold for 
umbral_distancia = [0, 50] # Threshold for the distance between distorted and theoretical pinholes
umbral_min = 700 # Minimum intensity to be detected as a pinhole

# Resultls parameter
# Contador de cuantos mapas/imagenes generar
# Si np.inf se generan para todas las simulaciones
UndistImg_cont=10
DistMap_cont=10
ErrorMap_cont=10
RotMap_cont=10
KValuesMap_cont=10
medir_rot=True #Si False se coge el angulo real computado de la simulacion
                #Si True el algoritmo mide el anglo de la placa 
medir_spacing=True #Si True se el algoritmo calcula spacing
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


