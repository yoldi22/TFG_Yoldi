from Simulation.ImageSimulator import GenerateImage
from Utils.utils import Save_Simulation, Init_Simulation_Directories
import time
import pandas as pd

# Paths
MainCarpet = r"DatosResultados/Simulaciones_tfg/prueba7"

# Simulation parameters
px = 5.5e-6 # pixelsize in meters
sen_dim = [3072, 4096] # sensor dimension [dimy,dimx]
radio = 10 # pinholes radius in pixels
num_imagenes = 100 # number of simulated images
n_ph = [15,20] # number of pinholes in each dimension
            # if None, aleatory
centro_optico = [sen_dim[1]/2 + 258,sen_dim[0]/2] #position in px of the optic center

# Errors parameters
delta_w = 0 * px # displacement of the plate center
delta_h = 0 * px # in each dimension

delta_cx = 0 * px # displacement of the optic center
delta_cy = 0 * px # in each dimension

delta_degree = 0 # rotation of the plate

kpx_max = 5 # Max distortion (in px)
kpx_min = 5 # Min distortion (in px)

I_max = 3000 # I max pinhole
I_min = 3000 # I min pinhole

# Initialization of the simulation
paths = Init_Simulation_Directories(MainCarpet)
DataCsv_path = paths[2]
resultados_data = []

total_start = time.perf_counter()  # tiempo total del bucle

for i in range(num_imagenes):

    img_start = time.perf_counter()  # tiempo por imagen

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
        I_min=I_min)

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

total_time = time.perf_counter() - total_start
print(f"Total processing time: {total_time:.2f}s for {i+1} images")

# Se cargan los resultados al CSV de resultados
df_res = pd.DataFrame(resultados_data)
df_res.to_csv(DataCsv_path, mode='a', header=False, index=False, sep='\t')