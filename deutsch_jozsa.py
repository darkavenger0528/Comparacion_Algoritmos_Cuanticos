"""
deutsch_jozsa.py
-----------------
Compara la evaluación clásica por fuerza bruta (para determinar si una
función booleana es constante o balanceada) contra el algoritmo de
Deutsch-Jozsa simulado en Qiskit, para un número creciente de bits de
entrada.

CSV de salida en formato ancho: una fila por tamaño de problema, con
columnas separadas clasico_ / dj_. La columna `dj_backend_real` es lo
que Qiskit Aer reporta que usó de verdad (CPU o GPU) en sus metadatos,
independiente de lo que se haya pedido con --device.
"""

import argparse
import csv
import os
import random
import sys

from resource_monitor import ResourceMonitor, gpu_monitoreo_disponible

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------
# Función oráculo compartida (balanceada, tipo producto punto)
# ---------------------------------------------------------------------

def generar_secreto(n_bits):
    s = random.randint(1, 2 ** n_bits - 1)
    return format(s, f"0{n_bits}b")


def f_balanceada(x_bits, secreto):
    return sum(int(a) * int(b) for a, b in zip(x_bits, secreto)) % 2


# ---------------------------------------------------------------------
# Versión clásica: fuerza bruta
# ---------------------------------------------------------------------

def evaluar_clasico(n_bits, secreto):
    resultados = set()
    for i in range(2 ** n_bits):
        x_bits = format(i, f"0{n_bits}b")
        resultados.add(f_balanceada(x_bits, secreto))
    return "balanceada" if len(resultados) == 2 else "constante"


def ejecutar_clasico(n_bits):
    secreto = generar_secreto(n_bits)
    with ResourceMonitor() as mon:
        evaluar_clasico(n_bits, secreto)
    return mon.summary()


# ---------------------------------------------------------------------
# Versión cuántica: Deutsch-Jozsa
# ---------------------------------------------------------------------

def construir_oraculo_dj(n_bits, secreto):
    qc = QuantumCircuit(n_bits + 1)
    for i, bit in enumerate(reversed(secreto)):
        if bit == "1":
            qc.cx(i, n_bits)
    return qc


def construir_circuito_dj(n_bits, secreto):
    qc = QuantumCircuit(n_bits + 1, n_bits)
    qc.x(n_bits)
    qc.h(range(n_bits + 1))
    qc.compose(construir_oraculo_dj(n_bits, secreto), inplace=True)
    qc.h(range(n_bits))
    qc.measure(range(n_bits), range(n_bits))
    return qc


def ejecutar_cuantico(n_bits, device="CPU"):
    secreto = generar_secreto(n_bits)
    simulador = AerSimulator(device=device)

    backend_real = None
    with ResourceMonitor() as mon:
        qc = construir_circuito_dj(n_bits, secreto)
        resultado = simulador.run(qc, shots=1024).result()
        resultado.get_counts()
        try:
            backend_real = resultado.results[0].metadata.get("device")
        except Exception:
            backend_real = "desconocido"

    return mon.summary(), backend_real


# ---------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Deutsch-Jozsa vs fuerza bruta clásica")
    parser.add_argument("--min-bits", type=int, default=2)
    parser.add_argument("--max-bits", type=int, default=18)
    parser.add_argument("--device", choices=["CPU", "GPU"], default="CPU")
    parser.add_argument("--carpeta", default="csv", help="Carpeta donde se guarda el CSV (se crea si no existe)")
    parser.add_argument("--salida", default="resultados_deutsch_jozsa.csv")
    args = parser.parse_args()

    os.makedirs(args.carpeta, exist_ok=True)
    ruta_salida = os.path.join(args.carpeta, args.salida)

    gpu_ok, gpu_err = gpu_monitoreo_disponible()
    if args.device == "GPU" and not gpu_ok:
        print(
            f"[AVISO] Pediste --device GPU pero el monitoreo de GPU (pynvml) "
            f"no se pudo inicializar: {gpu_err}. Revisa 'dj_backend_real' para "
            f"confirmar qué usó Qiskit realmente.",
            file=sys.stderr,
        )

    filas = []
    for n in range(args.min_bits, args.max_bits + 1):
        print(f"[Deutsch-Jozsa] n_bits={n}", file=sys.stderr)

        stats_clasico = ejecutar_clasico(n)
        stats_cuantico, backend_real = ejecutar_cuantico(n, device=args.device)

        print(f"    -> backend solicitado: {args.device} | backend real (Qiskit): {backend_real}", file=sys.stderr)

        fila = {"n_bits": n}
        for k, v in stats_clasico.items():
            fila[f"clasico_{k}"] = v
        for k, v in stats_cuantico.items():
            fila[f"dj_{k}"] = v
        fila["dj_backend_solicitado"] = args.device
        fila["dj_backend_real"] = backend_real
        filas.append(fila)

    campos = list(filas[0].keys())
    with open(ruta_salida, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(filas)

    print(f"Resultados guardados en {ruta_salida}")


if __name__ == "__main__":
    main()
