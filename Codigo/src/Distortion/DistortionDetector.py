import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA
from skimage.measure import label, regionprops
from scipy.spatial import KDTree

def Center_Detector(
        img,
        box_dim=50,
        ventana=100,
        umbral_max=0.4,
        umbral_min=500
):
    """
    Detecta centros con máxima precisión:
    - Usa maximum_filter para localizar picos.
    - Elimina ruido estudiando vecinos brillantes.
    - Calcula centroides con umbrales adaptativos.

    Parámetros:
    - img: imagen 2D.
    - box_dim: tamaño del filtro de máximos.
    - ventana: tamaño de la ventana centrada en cada pico.
    - umbral_max: qué tan brillante debe ser el umbral relativo 
        al máximo de la ventana.
    - umbral_min: intensidad mínima para considerar un pico.

    Retorna:
    Para visualización:
    - candidatos: Picos detectados por el filtro de máximos y que 
        superen el umbral_min
    - candidatos_filt: candidatos filtrados: ruido eliminado
    - subimg_: ventana (la de un pinhole medio)
    - binaria_: ventana subimg_ en binario después de pasarle el 
        umbral adaptativo
    - centroide_local_: centro de la ventana subimg_ relativa a la 
        ventana
    Para medición:
    - centros: centroides de los pinholes finales
    """
    # Detección inicial de máximos
    maximos = (img == maximum_filter(img, size=box_dim)) & (img > umbral_min)
    i, j = np.where(maximos)
    candidatos = np.column_stack((j, i))  # (x, y)
    candidatos_filt = []

    # Parámetros para diferenciar pinholes de ruido
    min_vecinos_brillantes = 0
    umbral_vecinos = 0.4

    # Filtrado conjunto 
    for ix, iy in candidatos:

        brillo_centro = img[iy, ix]
        vecinos_brillantes = 0

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                ny, nx = iy + dy, ix + dx
                if 0 <= ny < img.shape[0] and 0 <= nx < img.shape[1]:
                    if img[ny, nx] > umbral_vecinos * brillo_centro:
                        vecinos_brillantes += 1

        if vecinos_brillantes < min_vecinos_brillantes:
            continue

        candidatos_filt.append([ix, iy])

    candidatos_filt = np.array(candidatos_filt)

    # Cálculo de centroides 
    centros = []
    offset = ventana // 2
    i = 0

    for ix, iy in candidatos_filt:

        y0 = max(iy - offset, 0)
        y1 = min(iy + offset + 1, img.shape[0])
        x0 = max(ix - offset, 0)
        x1 = min(ix + offset + 1, img.shape[1])

        subimg = img[y0:y1, x0:x1]

        # Umbral adaptativo
        Imax = subimg.max() 
        umbral = umbral_max * Imax
        mask = subimg >= umbral
        binaria = mask.astype(np.uint8) 
        
        # Dividir las regiones mayores al umbral
        etiquetas = label(binaria)
        regiones = regionprops(etiquetas, intensity_image=subimg)

        if not regiones:
            continue

        # Identificar la región del pinhole como la mayor (en área)
        region = max(regiones, key=lambda r: r.area)

        region_mask = etiquetas == region.label
        ys, xs = np.nonzero(region_mask)
        weights = subimg[ys, xs] - umbral

        cx = np.sum(xs * weights) / np.sum(weights)
        cy = np.sum(ys * weights) / np.sum(weights)

        centros.append([x0 + cx, y0 + cy])

        if i == len(candidatos_filt) // 2:
            subimg_ = subimg
            binaria_ = binaria
            centroide_local_ = (cx, cy)
        i += 1   
    centros = np.array(centros)

    return (
        candidatos,
        candidatos_filt,
        subimg_,
        binaria_,
        centroide_local_,
        centros)

def Center_Sort_Hungarian(
        centros_sindist,
        centros_dist
):
    #Calcular la matriz de distancias entre cada par de puntos
    dist_matrix = distance_matrix(centros_sindist, centros_dist)

    #Resolver el problema de asignación óptima
    fila_indices, col_indices = linear_sum_assignment(dist_matrix)
    
    return centros_sindist[fila_indices], centros_dist[col_indices]

def Theroretical_Centers(
        spacing, 
        n_ph,
        centros_dist,
        centros_dist_filtrados,
        centro_placa, 
        centro_optico,
        rot=None
):
    """
    Calcula centros de pinholes con espaciado entero uniforme en píxeles,
    asegurando el mismo espaciado en X e Y y centrado en la imagen.

    Parámetros:
        - spacing: espaciado entre pinholes en px
        - n_ph: [n_x, n_y] numero de pinholes en cada dimensión
        - centros_dist: centros de los pinholes en la imagen real (distorsionada)
        - centro_placa: centro de la placa de pinholes

    Retorna:
        - np.array shape (N, 2) con coordenadas (x_px, y_px) en enteros
    """
    # Con los centros detectados estimamos la rotación de la placa,
    # a menos que se especifique
    if rot is None:
        angulo, components, spacing_medido = Detect_Rotation_Spacing(
                        centros_dist,
                        centro_optico, 
                        spacing)
        angulo = 0
        eje1 = [np.cos(angulo), np.sin(angulo)]
        eje2 = [np.cos(angulo + np.pi / 2), np.sin(angulo + np.pi / 2)]
        components = np.array([eje1, eje2])
    else:
        angulo=-rot
        eje1 = [np.cos(angulo), np.sin(angulo)]
        eje2 = [np.cos(angulo + np.pi / 2), np.sin(angulo + np.pi / 2)]
        components = np.array([eje1, eje2])
        spacing_medido = spacing
    
    error_spacing = np.abs(spacing - spacing_medido)
    spacing = spacing_medido #Usar el medido (o no-->comentar esta linea)

    # Generamos coordenadas centradas en (0,0)
    n_x, n_y = np.array(n_ph) * 2 # Se dibujan el doble de pinholes para cubrir toda la imagen
    start_x = - n_x / 2 * spacing
    start_y = - n_y / 2 * spacing
    cx = np.arange(n_x) * spacing + start_x
    cy = np.arange(n_y) * spacing + start_y
    X, Y = np.meshgrid(cx, cy)
    centros = np.column_stack((X.ravel(), Y.ravel()))

    # Se rotan los centros "ideales"
    R = np.array([
    [np.cos(angulo), -np.sin(angulo)],
    [np.sin(angulo),  np.cos(angulo)]
    ])
    centros_rotados = centros @ R.T

    # Se busca el centro más cercano al centro óptico,
    # este será el pinhole más fiel a su posición ideal 
    # y el más valido para completar los demás pinholes,
    # a menos que se especifique un desplazamiento concreto
    dists = np.sqrt(np.sum((centros_dist_filtrados - centro_optico) ** 2, axis=1))
    idx = np.argmin(dists)
    centro_optimo = centros_dist_filtrados[idx]
    dist_centro_optimo = dists[idx]

    # Se alinean con la placa
    centros_alineados = centros_rotados + centro_optimo

    return centros_alineados, components, dist_centro_optimo, error_spacing

def Center_Proccessing(
        img, 
        n_ph, 
        spacing, #pixels
        umbral_min=70,
        centro_optico = None,
        rot=None
):
    # Indicador de simulación erronea
    simulacion_erronea = [False, ""]

    # Detección de centros de los pinholes
    data = Center_Detector(
        img=img,
        umbral_min=umbral_min) 
    centros_dist = data[5]

    # Si no se proporciona el centro óptico, se asume que coincide
    # con el centro de la imagen
    if centro_optico is None:
        sen_dim = img.shape
        centro_optico = (sen_dim[1] / 2, sen_dim[0] / 2) #px

    # Buscamos el centro de la placa
    cntrs = Plate_Center_Detector(centros_dist) #px
    centro_placa, centros_dist, centros_dist_filtrados = cntrs

    # Calculamos los centros teoricos
    centros_sindist, components, dist_centro_optimo, error_spacing = Theroretical_Centers(
                n_ph=n_ph,
                spacing=spacing, #px
                centros_dist=centros_dist,
                centros_dist_filtrados=centros_dist_filtrados,
                centro_placa=centro_placa,
                centro_optico=centro_optico, 
                rot=rot)
    centros_dist = centros_dist_filtrados

    #Centramos los pinholes en el centro óptico
    centros_dist = centros_dist - centro_optico  #px
    centros_sindist = centros_sindist - centro_optico

    #Ahora emparejamos los centros sin distorsionar con los distorsionados
    centros = Center_Sort_Hungarian(centros_sindist, centros_dist)
    centros_sindist, centros_dist = centros
    
    # Verificamos que se hayan encontrado suficientes pinholes
    n_sindist, n_dist = len(centros_sindist), len(centros_dist)
    if n_dist < n_sindist / 10:
        simulacion_erronea[0] = True #No hay sificientes pihnoles brillantes
        simulacion_erronea[1] = "pinholes insuficientes"

    data = [centros_sindist, centros_dist,
            simulacion_erronea, centro_optico,
            centro_placa, components,
            dist_centro_optimo, error_spacing]

    return data

def Detect_Rotation_Spacing(
        points,
        optical_center,
        spacing,
        tolerance=0.05
):
    """
    Refina la orientación de la rejilla basado en puntos cercanos al centro óptico
    
    Args:
        points: Array de puntos (N, 2)
        optical_center: Centro óptico (x, y)
        spacing: Espaciado esperado entre puntos
        tolerance: Tolerancia relativa para la distancia (default: 0.05 = 5%)
    
    Returns:
        angle: Ángulo promedio de rotación (radianes)
        components: Tupla con los vectores unitarios (v1, v2) de las direcciones
    """
    # 1. Filtrar puntos cercanos al centro óptico
    tree = KDTree(points)
    indices_near_center = tree.query_ball_point(optical_center, r=spacing * 3)
    near_points = points[indices_near_center]
    
    # 2. Calcular todos los pares de puntos
    n = len(near_points)
    vectors = []
    angles = []
    spacings = []
    for i in range(n):
        for j in range(i + 1, n):
            # Calcular distancia entre puntos
            dist = np.linalg.norm(near_points[i] - near_points[j])
            
            # Verificar si la distancia es similar al espaciado
            if abs(dist - spacing) < spacing * tolerance:
                # Calcular vector y ángulo
                vector = near_points[j] - near_points[i]
                angle = np.arctan2(vector[1], vector[0])
                if angle > 3 * np.pi / 4 or angle < - np.pi / 4  :
                    vector[0] = - vector[0]
                    vector[1] = - vector[1]
                    angle = np.arctan2(vector[1], vector[0])
                vectors.append(vector)
                angles.append(angle)
                spacings.append(dist)

    # Si no hay suficientes vectores, devolver valores por defecto
    if len(vectors) < 2:
        # Valores por defecto (sin rotación)
        return 0.0, (np.array([1.0, 0.0]), np.array([0.0, 1.0])), spacing
    
    # 3. Agrupar ángulos en dos direcciones perpendiculares
    angles = np.array(angles)
    vectors = np.array(vectors)
    spacings = np.array(spacings)

    # Agrupar usando KMeans con 2 clusters para direcciones
    from sklearn.cluster import KMeans
    
    # Representación angular para clustering (usamos coordenadas en círculo unitario)
    X = np.column_stack([np.cos(angles), np.sin(angles)])
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    labels = kmeans.labels_
    
    # Calcular ángulos promedio para cada grupo
    mean_angles = []
    mean_vectors = []
    for group in range(2):
        group_vectors = vectors[labels == group]
        
        # Calcular vector promedio del grupo
        mean_vector = np.mean(group_vectors, axis=0)
        mean_vector /= np.linalg.norm(mean_vector)
        
        # Calcular ángulo promedio
        mean_angle = np.arctan2(mean_vector[1], mean_vector[0])
        
        mean_angles.append(mean_angle)
        mean_vectors.append(mean_vector)

    # 4. Verificar que los ángulos sean aproximadamente perpendiculares
    angle_diff = abs(mean_angles[0] - mean_angles[1])
    perpendicular_diff = abs(angle_diff - np.pi/2)

    # Si la diferencia no es aproximadamente 90°, ajustar uno de los vectores
    if perpendicular_diff > np.pi/4:  # Más de 45° de desviación
        # Hacer el segundo vector perpendicular al primero
        v1 = mean_vectors[0]
        v2 = np.array([-v1[1], v1[0]])
        mean_vectors[1] = v2
        mean_angles[1] = np.arctan2(v2[1], v2[0])
    
    return np.min(mean_angles), (mean_vectors[0], mean_vectors[1]), np.mean(spacings)

def Calculation_k(
        centros_dist,
        centros_sindist
):
    
    # Calcular la distancia al centro
    r_dist = np.linalg.norm(centros_dist, axis=1)  
    r_sindist = np.linalg.norm(centros_sindist, axis=1)

    # Calcular los puntos de la regresión
    x_i = r_sindist ** 3 # pix^3
    y_i = r_dist - r_sindist # px

    # Calcular la distorsión mediante mínimos cuadrados y graficar
    k_medida = np.sum(x_i * y_i) / np.sum(x_i ** 2)

    return k_medida, x_i, y_i

def Validation(
        centros_dist,
        centros_sindist,
        simulacion,
        k_real,
        k_medida,
        umbral_dist,
        sen_dim,
        x_i,
        y_i
):
    simulacion_erronea = [False, ""]
    y_i_ajuste = x_i * k_medida #px^3 * px^-2
    residuos = y_i_ajuste - y_i #px

    # Intentamos mejorar la simulación eliminando las peores
    # medidad con el Método de Z-Score
    z_scores = (residuos - np.mean(residuos)) / np.std(residuos)
    mask = np.abs(z_scores) < umbral_dist  # solo puntos con |z| < 3

    # Se actualizan las listas
    x_filtrado = x_i[mask]
    y_filtrado = y_i[mask]
    centros_sindist = centros_sindist[mask]
    centros_dist = centros_dist[mask]
    n_sindist, n_dist = len(centros_sindist), len(centros_dist)
    
    # Verificamos que no se hayan eliminado demasiados pinholes
    if n_dist < 5:
        simulacion_erronea[0] = True #No hay sificientes pihnoles brillantes
        simulacion_erronea[1] = "Error en el filtrado Z-score"
        results = [
            k_medida,
            simulacion_erronea,
            [np.nan,np.nan,np.nan,np.nan],
            [centros_sindist, centros_dist],
            [x_i, x_filtrado],
            [y_i, y_filtrado]
        ]
        return results

    # Se vuelve a calcular la constante de distorsión después
    # del filtrado
    k_medida = np.sum(x_filtrado * y_filtrado) / np.sum(x_filtrado ** 2) # px^-2

    # después de recalcular k_medida se recalculan los residuos
    y_i_ajuste = x_filtrado * k_medida #px^3 * px^-2
    residuos = y_i_ajuste - y_filtrado #px

    # Se calcula el coeficiente para cada pinhole
    k_values = y_filtrado / x_filtrado

    # Se calculan errores del ajuste
    # RMSE
    RMSE = np.mean(np.sqrt((k_values - k_medida) ** 2))  #px

    # R^2
    R2 = 1 - np.sum(residuos ** 2) / np.sum((y_filtrado - np.mean(y_filtrado)) ** 2) #adimensional
    error_relativo = np.nan
    MDLD = np.nan
    
    if simulacion:

        # Se calculan distintas metricas

        # Error relativo del coeficiente
        error_relativo = np.abs((k_real - k_medida) / k_real) * 100 #adimensional

        # MDLD
        R = np.linalg.norm(centros_dist, axis=1)  # matriz de distancias al centro
        d_ground = 1 + k_real * (R ** 2)
        d_medida = 1 + k_values * (R ** 2)
        MDLD = np.mean(np.abs(d_medida - d_ground))

        # Se verifica la validez 
        if error_relativo > 5:
            # Indicador de medición erronea
            simulacion_erronea[0] = True
            simulacion_erronea[1] = "Error relativo excesivo"

    errores = [
        error_relativo,
        MDLD,
        RMSE,
        R2
    ]

    results = [
        k_medida,
        simulacion_erronea,
        errores,
        [centros_sindist, centros_dist],
        [x_i, x_filtrado],
        [y_i, y_filtrado]
    ]

    return results

def Plate_Center_Detector(
        centros,
        n_vecinos=4,
        umbral=5
):
    N = len(centros)
    distancias = distance_matrix(centros, centros)
    np.fill_diagonal(distancias, np.inf)  # evitar distancia a sí mismo

    vecinos = {}
    distancias_sum = np.empty(N)

    for i in range(N):
        indices_vecinos = np.argpartition(distancias[i], n_vecinos)[:n_vecinos]
        vecinos[i] = indices_vecinos.tolist()
        distancias_sum[i] = np.sum(distancias[i, indices_vecinos])

    # Detectar centro
    centro_idx = np.argmin(distancias_sum)
    centro = centros[centro_idx]

    # Evaluar distancias a vecinos más cercanos al centro
    vecinos_centro = vecinos[centro_idx]
    dist_centro = distancias[centro_idx, vecinos_centro]
    dist_min = np.min(dist_centro)

    # Vecinos cuya distancia está dentro del umbral 
    vecinos_centro = np.array(vecinos_centro)
    vecinos_cercanos = vecinos_centro[dist_centro <= umbral + dist_min]

    # Marcar índices a eliminar (centro + vecinos cercanos)
    indices_a_eliminar = set(vecinos_cercanos.tolist())
    indices_a_eliminar.add(centro_idx)

    # Filtrar centros
    mask = np.array([i not in indices_a_eliminar for i in range(N)])
    centros_filtrados = centros[mask]

    return centro, centros, centros_filtrados

def Distortion_Detector(
        img,
        n_ph,
        sen_dim,
        centro_optico,
        spacing, #px
        umbral_dist,
        simulacion,
        umbral_distancia,   #[min_dist, max_dist] px
        k_real=np.nan, #px
        umbral_min=70,
        rot=None
):
  
    # Se detectan los centros
    data = Center_Proccessing(
                    img=img,
                    n_ph=n_ph,
                    spacing=spacing,
                    centro_optico=centro_optico,
                    umbral_min=umbral_min,
                    rot=rot)
    centros_sindist = data[0] #px relativos a centro optico
    centros_dist = data[1] #px relativos a centro optico
    simulacion_erronea = data[2]
    centro_optico = data[3]
    centro_placa = data[4]
    components = data[5]
    dist_centro_optimo = data[6]
    error_spacing = data[7]

    # Se verifica que la detección ha sido correcta
    if simulacion_erronea[0]:
        results = [ np.nan, 
                    [np.nan,np.nan,np.nan,np.nan],
                    simulacion_erronea,
                    centros_sindist, 
                    centros_dist, 
                    centro_placa, 
                    components, 
                    np.nan,
                    np.nan, 
                    dist_centro_optimo,
                    error_spacing,
                    centro_optico]
        return results

    # Se eliminan los centros que se hayan desplazado fuera 
    # de los límites deseados
    desplazamientos = centros_dist - centros_sindist
    distancias = np.linalg.norm(desplazamientos, axis=1)

    # Crear máscara de puntos válidos
    mask =(distancias >= umbral_distancia[0]) & (distancias <= umbral_distancia[1])
    centros_sindist, centros_dist = centros_sindist[mask], centros_dist[mask]

    # Se calcula la constante de distorsión
    k_medida, x_i, y_i = Calculation_k(centros_dist=centros_dist,
                            centros_sindist=centros_sindist)

    # Se validan los resultados si provienen de una simulación
    results = Validation(
                centros_dist=centros_dist,
                centros_sindist=centros_sindist,
                simulacion=simulacion,
                k_medida=k_medida,
                k_real=k_real,
                umbral_dist=umbral_dist,
                sen_dim=sen_dim,
                x_i=x_i,
                y_i=y_i)       
    k_medida = results[0]
    simulacion_erronea = results[1]
    errores = results[2]
    centros_sindist = results[3][0]
    centros_dist = results[3][1]
    x_i = results[4]
    y_i = results[5]
        
    results = [ k_medida, 
                errores,
                simulacion_erronea,
                centros_sindist, 
                centros_dist, 
                centro_placa, 
                components, 
                x_i,
                y_i,
                dist_centro_optimo,
                error_spacing,
                centro_optico]

    return results

def Displacement_Detector(
        img_undist,
        img_ideal,
        umbral=20,  
        umbral_min=70
):
    data = Center_Detector(
                    img_undist,
                    umbral_min=umbral_min)
    centros_undist = data[5]
    data = Center_Detector(
                    img_ideal,
                    umbral_min=umbral_min)
    centros_ideal = data[5]
    
    # Emparejar centros (Hungarian)
    centros_ideal, centros_undist = Center_Sort_Hungarian(centros_ideal, centros_undist)

    # Calcular desplazamientos
    desplazamientos = centros_undist - centros_ideal
    distancias = np.linalg.norm(desplazamientos, axis=1)

    # Filtrar errores de emparejamiento
    mask = (distancias < umbral)

    return centros_ideal[mask], desplazamientos[mask], distancias[mask]




        