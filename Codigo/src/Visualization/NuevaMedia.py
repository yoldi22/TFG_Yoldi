import pandas as pd
import numpy as np


def calcular_media_filtrada(
    simulaciones_path,
    errors_path,
    results_path,
    output_path,
    k_threshold=5
):
    # Leer archivos con separador TAB
    sim_df = pd.read_csv(simulaciones_path, sep="\t")
    err_df = pd.read_csv(errors_path, sep="\t")
    res_df = pd.read_csv(results_path, sep="\t")

    # Limpiar nombres de columnas
    sim_df.columns = sim_df.columns.str.strip()
    err_df.columns = err_df.columns.str.strip()
    res_df.columns = res_df.columns.str.strip()

    # Convertir Number a numérico
    err_df["Number"] = pd.to_numeric(err_df["Number"], errors="coerce")
    res_df["Number"] = pd.to_numeric(res_df["Number"], errors="coerce")

    # Añadir columna Number al dataframe de simulaciones
    sim_df = sim_df.reset_index().rename(columns={"index": "Number"})
    sim_df["Number"] = pd.to_numeric(sim_df["Number"], errors="coerce")

    # Merge de los tres CSV
    merged = pd.merge(res_df, err_df, on="Number", how="inner")
    merged = pd.merge(
        merged,
        sim_df[["Number", "k_real(m^-2)"]],
        on="Number",
        how="inner"
    )

    # Filtrar |k_real| > threshold
    filtered = merged[np.abs(merged["k_real(m^-2)"]) > k_threshold]

    # Columnas para media
    columnas_media = [
        "RMSE(m)",
        "MDLD",
        "R2",
        "DistC_o(px)",
        "R_Error"
    ]

    # Calcular medias solo si existen
    columnas_existentes = [c for c in columnas_media if c in filtered.columns]
    medias = filtered[columnas_existentes].mean()

    # Crear fila MEDIA
    fila_media = pd.DataFrame([medias])
    fila_media.index = ["MEDIA"]

    # Concatenar filas filtradas + MEDIA
    final_df = pd.concat([filtered, fila_media], ignore_index=False)

    # Guardar CSV final
    final_df.to_csv(output_path, sep="\t", index=True)
    
path = r"DatosResultados\Simulaciones_tfg"
calcular_media_filtrada(
    simulaciones_path=path+r"\prueba6\Simulaciones_data.csv",
    errors_path=path+r"\prueba8.2\Errors\Errors.csv",
    results_path=path+r"\prueba8.2\DataResults.csv",
    output_path=r"Resultados_filtrados.csv",
    k_threshold=5
)
