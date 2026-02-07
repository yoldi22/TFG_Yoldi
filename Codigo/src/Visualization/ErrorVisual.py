import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def AnalyzeErrorsVsK(
        results_csv_path,
        errors_csv_path,
        save_path=None
):
    """
    Analiza las métricas de error en función de k_real y genera un gráfico.

    Parámetros
    ----------
    results_csv_path : str
        Ruta al CSV de resultados globales (RMSE, R2, k).
    errors_csv_path : str
        Ruta al CSV de errores geométricos (MDLD, error relativo).
    save_path : str | None
        Si se indica, guarda la figura en esta ruta.
    """

    # =========================
    # Cargar CSVs
    # =========================
    df_res = pd.read_csv(results_csv_path, sep="\t")
    df_err = pd.read_csv(errors_csv_path, sep="\t")

    # =========================
    # Forzar columnas numéricas
    # (elimina filas tipo "MEDIA")
    # =========================
    numeric_cols_res = ["Measured_k(m^-2)", "RMSE(m)", "R2"]
    numeric_cols_err = ["MDLD", "R_Error"]

    for col in numeric_cols_res:
        df_res[col] = pd.to_numeric(df_res[col], errors="coerce")

    for col in numeric_cols_err:
        df_err[col] = pd.to_numeric(df_err[col], errors="coerce")

    df_res = df_res.dropna(subset=numeric_cols_res)
    df_err = df_err.dropna(subset=numeric_cols_err)

    # =========================
    # Variables
    # =========================
    k_real = df_res["Measured_k(m^-2)"].to_numpy()
    rmse   = df_res["RMSE(m)"].to_numpy()
    r2     = df_res["R2"].to_numpy()
    mdld   = df_err["MDLD"].to_numpy()
    relerr = df_err["R_Error"].to_numpy()

    # =========================
    # Gráfico
    # =========================
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(k_real, rmse, "o-", label="RMSE")
    ax1.plot(k_real, 1 - r2, "s-", label="1 - R²")
    ax1.plot(k_real, mdld, "^-", label="MDLD")
    ax1.set_xlabel(r"$k_{\mathrm{real}}\,(\mathrm{m}^{-2})$")
    ax1.set_ylabel("Error geométrico")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(k_real, relerr, "x--", color="black", label="Error relativo")
    ax2.set_ylabel("Error relativo")

    fig.legend(loc="upper left")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()

results_csv_path = r'Simulaciones_tfg/TODOFIJO_kreal/Total.csv'
errors_csv_path = r'Simulaciones_tfg/TODOFIJO_kreal/Errors/Etotal.csv'
save_path = r'Simulaciones_tfg/TODOFIJO_kreal/errores_vs_k.png'
AnalyzeErrorsVsK(
    results_csv_path=results_csv_path,
    errors_csv_path=errors_csv_path,
    save_path=save_path)