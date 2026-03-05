from Simulation.ImageSimulator import GenerateImage
from Utils.utils import Save_Simulation, Init_Simulation_Directories
import pandas as pd

# Paths
MainCarpet = "" # Nombre de la carpeta donde se almacenan todos los resutlados

# Parámetros de la simulación
px = # pixelsize en metros
sen_dim = # Dimensiones del sensor [dimy,dimx]
radio = # Radio aproximado de los pinholes
num_imagenes =  # Número de imágenes simuladas
n_ph =  # Número de pinholes en cada dimensión [ny, nx], si None: aleatoria
centro_optico = #Posición del centro óptico [y,x]

# Errores o variables
delta_w = # Incertidumbre en el desplazamiento en cada dimensión (en metros)
delta_h = 

delta_cx = # Incertidumbre del centro óptico (en metros)
delta_cy = 

delta_degree =  # Incertidumbre en la rotación de la placa

# El algoritmo introduce una distorsión aleatoria entre kpx_max y kpx_min
# si kpx_max=kpxmin se introduce la misma en todas las simulaciones
kpx_max = # Desplazamiento del pinhole más alejado del centro (en px) 
kpx_min =

# Los pinholes tendrán picos de intensidad máxima aleatoria 
# entre I_min e I_max (0, 4095) 
I_max =  
I_min = 

# Cantidad de pinholes dibujados 
# 0: ningún pinhole dibujado
# 1: Todos los pinholes dibujados
I_oscuro = 



# Initialization of the simulation
paths = Init_Simulation_Directories(MainCarpet)
DataCsv_path = paths[2]
resultados_data = []

for i in range(num_imagenes):

    results = GenerateImage(
        sen_dim=sen_dim,
        n_ph=n_ph,
        radio=radio,
        delta_cx=delta_cx,
        delta_cy=delta_cy,
        delta_h=delta_h,
        delta_w=delta_w,
        delta_degree=delta_degree,
        px=px,
        centro_optico=centro_optico,
        kpx_max=kpx_max,
        kpx_min=kpx_min,
        I_max=I_max,
        I_min=I_min,
        I_oscuro=I_oscuro)

    Save_Simulation(
        IdealImg=results[0],
        DistImg=results[1],
        px=px,
        i=i,
        paths=paths)

    # Se guardan los datos del CSV
    resultados_data.append({
        "Nombre_ideal"      : f"img_ideal_{i:04d}.fits",
        "Nombre_dist"       : f"img_dist_{i:04d}.fits",
        "k_real(m^-2)"      : results[2] / (px ** 2), # de pixeles a metros
        "n_ph_x"            : results[3][0],
        "n_ph_y"            : results[3][1],
        "Spacing(m)"        : results[4] * px, # de pixeles a metros
        "Cntr_optico_x(px)" : results[5][0], # pixeles
        "Cntr_optico_y(px)" : results[5][1], # pixeles
        "Rot(Rad)"          : results[6], # Degrees
        "SenDim_x"          : sen_dim[1],
        "SenDim_y"          : sen_dim[0],
        "PX(m)"             : px})

# Se cargan los resultados al CSV de resultados
df_res = pd.DataFrame(resultados_data)
df_res.to_csv(DataCsv_path, mode='a', header=False, index=False, sep='\t')