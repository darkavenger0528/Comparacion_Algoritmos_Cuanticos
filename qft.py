"""
qft.py
------
Compara la Transformada Rápida de Fourier (FFT) clásica (NumPy) contra
la Transformada Cuántica de Fourier (QFT) simulada en Qiskit, para
vectores/estados de tamaño N = 2^n_qubits.

CSV de salida en formato ancho: una fila por tamaño de problema, con
columnas separadas clasico_ / qft_. La columna `qft_backend_real` es lo
que Qiskit Aer reporta que usó de verdad (CPU o GPU) en sus metadatos.
"""

import argparse
import csv
import os
import sys

import numpy as np

from resource_monitor import ResourceMonitor, gpu_monitoreo_disponible

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------
# Versión clásica: FFT
# ---------------------------------------------------------------------

def ejecutar_clasico(n_qubits):
    N = 2 ** n_qubits
    vector = np.random.random(N) + 1j * np.random.random(N)
    with ResourceMonitor() as mon:
        np.fft.fft(vector)
    return mon.summary()


# ---------------------------------------------------------------------
# Versión cuántica: QFT
# ---------------------------------------------------------------------

def ejecutar_cuantico(n_qubits, device="CPU"):
    simulador = AerSimulator(method="statevector", device=device)

    backend_real = None
    with ResourceMonitor() as mon:
        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits))
        qc.append(QFT(n_qubits), range(n_qubits))
        qc.save_statevector()
        qc = transpile(qc, simulador)
        resultado = simulador.run(qc).result()
        resultado.get_statevector()
        try:
            backend_real = resultado.results[0].metadata.get("device")
        except Exception:
            backend_real = "desconocido"

    return mon.summary(), backend_real


# ---------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="QFT cuántica vs FFT clásica")
    parser.add_argument("--min-qubits", type=int, default=2)
    parser.add_argument("--max-qubits", type=int, default=18)
    parser.add_argument("--device", choices=["CPU", "GPU"], default="CPU")
    parser.add_argument("--carpeta", default="csv", help="Carpeta donde se guarda el CSV (se crea si no existe)")
    parser.add_argument("--salida", default="resultados_qft.csv")
    args = parser.parse_args()

    os.makedirs(args.carpeta, exist_ok=True)
    ruta_salida = os.path.join(args.carpeta, args.salida)

    gpu_ok, gpu_err = gpu_monitoreo_disponible()
    if args.device == "GPU" and not gpu_ok:
        print(
            f"[AVISO] Pediste --device GPU pero el monitoreo de GPU (pynvml) "
            f"no se pudo inicializar: {gpu_err}. Revisa 'qft_backend_real' para "
            f"confirmar qué usó Qiskit realmente.",
            file=sys.stderr,
        )

    filas = []
    for n in range(args.min_qubits, args.max_qubits + 1):
        print(f"[QFT] n_qubits={n} (N={2 ** n})", file=sys.stderr)

        stats_clasico = ejecutar_clasico(n)
        stats_cuantico, backend_real = ejecutar_cuantico(n, device=args.device)

        print(f"    -> backend solicitado: {args.device} | backend real (Qiskit): {backend_real}", file=sys.stderr)

        fila = {"n_qubits": n, "N": 2 ** n}
        for k, v in stats_clasico.items():
            fila[f"clasico_{k}"] = v
        for k, v in stats_cuantico.items():
            fila[f"qft_{k}"] = v
        fila["qft_backend_solicitado"] = args.device
        fila["qft_backend_real"] = backend_real
        filas.append(fila)

    campos = list(filas[0].keys())
    with open(ruta_salida, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(filas)

    print(f"Resultados guardados en {ruta_salida}")


if __name__ == "__main__":
    main()
