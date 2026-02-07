
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def AnalyzeErrorsVsK(
        results_csv_path,
        errors_csv_path,
        save_path=None,
        log_relerr=False
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
    relerr = np.clip(relerr, None, 30)
    
    # =========================
    # Gráfico
    # =========================
    # ======================
    # 1️⃣ RMSE vs k
    # ======================
    plt.figure(figsize=(7, 5))
    plt.plot(k_real, rmse, "+")
    plt.xlabel(r"$k_{\mathrm{real}}\;(\mathrm{m}^{-2})$")
    plt.ylabel("RMSE (m)")
    plt.title("RMSE en función de $k_{real}$")
    plt.grid(True)
    if save_path:
        plt.savefig(f"{save_path}/RMSE_vs_k.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ======================
    # 2️⃣ R² vs k
    # ======================
    plt.figure(figsize=(7, 5))
    plt.plot(k_real, r2, "+")
    plt.xlabel(r"$k_{\mathrm{real}}\;(\mathrm{m}^{-2})$")
    plt.ylabel(r"$R^2$")
    plt.title(r"$R^2$ en función de $k_{real}$")
    plt.grid(True)
    if save_path:
        plt.savefig(f"{save_path}/R2_vs_k.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ======================
    # 3️⃣ Error relativo vs k
    # ======================
    plt.figure(figsize=(7, 5))
    plt.plot(k_real, relerr, "+")
    plt.xlabel(r"$k_{\mathrm{real}}\;(\mathrm{m}^{-2})$")
    plt.ylabel("Error relativo")

    if log_relerr:
        plt.yscale("log")
        plt.title("Error relativo (escala logarítmica)")
    else:
        plt.title("Error relativo")

    plt.grid(True, which="both")
    if save_path:
        plt.savefig(f"{save_path}/RelativeError_vs_k.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ======================
    # 4️⃣ MDLD vs k
    # ======================
    plt.figure(figsize=(7, 5))
    plt.plot(k_real, mdld, "+")
    plt.xlabel(r"$k_{\mathrm{real}}\;(\mathrm{m}^{-2})$")
    plt.ylabel("MDLD")
    plt.title("MDLD en función de $k_{real}$")
    plt.grid(True)
    if save_path:
        plt.savefig(f"{save_path}/MDLD_vs_k.png", dpi=300, bbox_inches="tight")
    plt.show()

results_csv_path = r'Simulaciones_tfg/TODOFIJO_kreal/Total.csv'
errors_csv_path = r'Simulaciones_tfg/TODOFIJO_kreal/Errors/Etotal.csv'
save_path = r'Simulaciones_tfg/TODOFIJO_kreal'
AnalyzeErrorsVsK(
    results_csv_path=results_csv_path,
    errors_csv_path=errors_csv_path,
    save_path=save_path)

