"""
graficar_resultados.py
-----------------------
Lee los CSV generados por grover.py, deutsch_jozsa.py, qft.py y shor.py
(por defecto desde la carpeta 'csv/') y genera automáticamente las
gráficas comparativas de tiempo, RAM y uso de GPU para cada algoritmo,
guardándolas como imágenes PNG en la carpeta 'graficas/'.

No hace falta tener los 8 archivos: el script salta silenciosamente
los algoritmos cuyos CSV no encuentre.

Uso:
    venv/bin/python graficar_resultados.py
    venv/bin/python graficar_resultados.py --csv otra_carpeta --salida otras_graficas
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")  # no necesita pantalla, solo guarda archivos
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Utilidades de lectura
# ---------------------------------------------------------------------

def leer_csv(ruta):
    if not os.path.exists(ruta):
        return None
    with open(ruta, newline="") as f:
        return list(csv.DictReader(f))


def col_float(filas, nombre):
    """Convierte una columna a floats; None/vacíos quedan como None."""
    out = []
    for fila in filas:
        val = fila.get(nombre, "")
        try:
            out.append(float(val))
        except (ValueError, TypeError):
            out.append(None)
    return out


# ---------------------------------------------------------------------
# Gráficas para Grover / Deutsch-Jozsa / QFT (formato ancho)
# ---------------------------------------------------------------------

def graficar_algoritmo_ancho(carpeta_csv, carpeta_salida, nombre_archivo, prefijo_cuantico,
                              columna_x, etiqueta_x, titulo_amigable):
    """
    nombre_archivo: base sin sufijo, ej. 'grover' -> busca grover_cpu.csv y grover_gpu.csv
    prefijo_cuantico: prefijo de columnas del algoritmo cuántico, ej. 'grover', 'dj', 'qft'
    """
    filas_cpu = leer_csv(os.path.join(carpeta_csv, f"{nombre_archivo}_cpu.csv"))
    filas_gpu = leer_csv(os.path.join(carpeta_csv, f"{nombre_archivo}_gpu.csv"))

    if filas_cpu is None and filas_gpu is None:
        print(f"[omitido] No se encontraron CSV para '{nombre_archivo}' en {carpeta_csv}")
        return

    referencia = filas_cpu if filas_cpu is not None else filas_gpu
    x = col_float(referencia, columna_x)

    # --- Gráfica 1: tiempo de ejecución ---
    plt.figure(figsize=(8, 5))
    if filas_cpu is not None:
        plt.plot(x, col_float(filas_cpu, "clasico_tiempo_segundos"), marker="o", label="Clásico")
        plt.plot(x, col_float(filas_cpu, f"{prefijo_cuantico}_tiempo_segundos"), marker="o", label=f"{titulo_amigable} (CPU)")
    if filas_gpu is not None:
        plt.plot(x, col_float(filas_gpu, f"{prefijo_cuantico}_tiempo_segundos"), marker="o", label=f"{titulo_amigable} (GPU)")
    plt.yscale("log")
    plt.xlabel(etiqueta_x)
    plt.ylabel("Tiempo (segundos, escala log)")
    plt.title(f"{titulo_amigable}: tiempo de ejecución")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    ruta = os.path.join(carpeta_salida, f"{nombre_archivo}_tiempo.png")
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"[ok] {ruta}")

    # --- Gráfica 2: RAM ---
    plt.figure(figsize=(8, 5))
    if filas_cpu is not None:
        plt.plot(x, col_float(filas_cpu, "clasico_ram_maximo_mb"), marker="o", label="Clásico")
        plt.plot(x, col_float(filas_cpu, f"{prefijo_cuantico}_ram_maximo_mb"), marker="o", label=f"{titulo_amigable} (CPU)")
    if filas_gpu is not None:
        plt.plot(x, col_float(filas_gpu, f"{prefijo_cuantico}_ram_maximo_mb"), marker="o", label=f"{titulo_amigable} (GPU)")
    plt.xlabel(etiqueta_x)
    plt.ylabel("RAM máxima (MB)")
    plt.title(f"{titulo_amigable}: uso de RAM")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    ruta = os.path.join(carpeta_salida, f"{nombre_archivo}_ram.png")
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"[ok] {ruta}")

    # --- Gráfica 3: uso de GPU (solo si hay datos de la corrida GPU) ---
    if filas_gpu is not None:
        gpu_mem = col_float(filas_gpu, f"{prefijo_cuantico}_gpu_memoria_maxima_mb")
        gpu_util = col_float(filas_gpu, f"{prefijo_cuantico}_gpu_uso_maximo_pct")
        if any(v is not None for v in gpu_mem) or any(v is not None for v in gpu_util):
            fig, ax1 = plt.subplots(figsize=(8, 5))
            ax1.plot(x, gpu_mem, marker="o", color="tab:blue", label="Memoria GPU (MB)")
            ax1.set_xlabel(etiqueta_x)
            ax1.set_ylabel("Memoria GPU máxima (MB)", color="tab:blue")
            ax1.tick_params(axis="y", labelcolor="tab:blue")

            ax2 = ax1.twinx()
            ax2.plot(x, gpu_util, marker="s", color="tab:red", label="Uso GPU (%)")
            ax2.set_ylabel("Uso de GPU máximo (%)", color="tab:red")
            ax2.tick_params(axis="y", labelcolor="tab:red")

            plt.title(f"{titulo_amigable}: uso de GPU (corrida GPU)")
            fig.tight_layout()
            ruta = os.path.join(carpeta_salida, f"{nombre_archivo}_gpu.png")
            plt.savefig(ruta, dpi=150)
            plt.close()
            print(f"[ok] {ruta}")
        else:
            print(f"[aviso] '{nombre_archivo}': no hay datos de GPU medidos (pynvml/nvidia-ml-py no activo en esa corrida)")


# ---------------------------------------------------------------------
# Gráficas para Shor (formato largo: columna 'tipo')
# ---------------------------------------------------------------------

def graficar_shor(carpeta_csv, carpeta_salida):
    filas_cpu = leer_csv(os.path.join(carpeta_csv, "shor_cpu.csv"))
    filas_gpu = leer_csv(os.path.join(carpeta_csv, "shor_gpu.csv"))

    if filas_cpu is None and filas_gpu is None:
        print(f"[omitido] No se encontraron CSV de Shor en {carpeta_csv}")
        return

    referencia = filas_cpu if filas_cpu is not None else filas_gpu
    clasicas = [f for f in referencia if f["tipo"] == "clasico"]
    x_clasico = col_float(clasicas, "parametro")
    y_clasico = col_float(clasicas, "tiempo_segundos")

    plt.figure(figsize=(8, 5))
    plt.plot(x_clasico, y_clasico, marker="o", label="División de prueba (clásico)")
    plt.yscale("log")
    plt.xlabel("Bits del número factorizado")
    plt.ylabel("Tiempo (segundos, escala log)")
    plt.title("Shor: división de prueba clásica vs. tamaño del número")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    ruta = os.path.join(carpeta_salida, "shor_clasico_tiempo.png")
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"[ok] {ruta}")

    plt.figure(figsize=(8, 5))
    if filas_cpu is not None:
        cuanticas_cpu = [f for f in filas_cpu if f["tipo"] == "shor"]
        plt.plot(col_float(cuanticas_cpu, "parametro"), col_float(cuanticas_cpu, "tiempo_segundos"),
                 marker="o", label="Shor N=15 (CPU)")
    if filas_gpu is not None:
        cuanticas_gpu = [f for f in filas_gpu if f["tipo"] == "shor"]
        plt.plot(col_float(cuanticas_gpu, "parametro"), col_float(cuanticas_gpu, "tiempo_segundos"),
                 marker="o", label="Shor N=15 (GPU)")
    plt.yscale("log")
    plt.xlabel("Qubits del registro de conteo (precisión)")
    plt.ylabel("Tiempo (segundos, escala log)")
    plt.title("Shor: circuito cuántico (N=15) vs. qubits de conteo")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    ruta = os.path.join(carpeta_salida, "shor_cuantico_tiempo.png")
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"[ok] {ruta}")


# ---------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Genera gráficas a partir de los CSV de resultados")
    parser.add_argument("--csv", default="csv", help="Carpeta donde están los CSV de entrada")
    parser.add_argument("--salida", default="graficas", help="Carpeta donde se guardan las gráficas (PNG)")
    args = parser.parse_args()

    os.makedirs(args.salida, exist_ok=True)

    graficar_algoritmo_ancho(args.csv, args.salida, "grover", "grover", "n_qubits", "n_qubits (N=2^n)", "Grover")
    graficar_algoritmo_ancho(args.csv, args.salida, "dj", "dj", "n_bits", "n_bits", "Deutsch-Jozsa")
    graficar_algoritmo_ancho(args.csv, args.salida, "qft", "qft", "n_qubits", "n_qubits (N=2^n)", "QFT")
    graficar_shor(args.csv, args.salida)

    print(f"\nListo. Revisa la carpeta '{args.salida}/' para ver todas las gráficas generadas.")


if __name__ == "__main__":
    main()
