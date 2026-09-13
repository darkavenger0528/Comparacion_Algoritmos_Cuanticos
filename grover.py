"""
grover.py
---------
Compara la búsqueda lineal clásica contra el algoritmo de Grover
simulado en Qiskit, para distintos tamaños de espacio de búsqueda
(N = 2^n_qubits).

El CSV de salida usa formato "ancho": una fila por tamaño de problema,
con columnas separadas con prefijo clasico_ / grover_, para que sea
fácil de leer y de graficar directamente sin filtrar filas.

Además, cada fila cuántica incluye la columna `grover_backend_real`,
que es el valor que Qiskit Aer reporta en los metadatos del resultado
sobre qué dispositivo usó de verdad (CPU o GPU) — esta es la fuente de
verdad definitiva, independiente de lo que herramientas externas como
btop puedan sugerir.
"""

import argparse
import csv
import math
import os
import random
import sys

from resource_monitor import ResourceMonitor, gpu_monitoreo_disponible

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------
# Versión clásica: búsqueda lineal
# ---------------------------------------------------------------------

def busqueda_lineal(lista, objetivo):
    for i, valor in enumerate(lista):
        if valor == objetivo:
            return i
    return -1


def ejecutar_clasico(n_qubits):
    N = 2 ** n_qubits
    lista = list(range(N))
    objetivo = -1  # fuerza el peor caso: recorrer toda la lista
    with ResourceMonitor() as mon:
        busqueda_lineal(lista, objetivo)
    return mon.summary()


# ---------------------------------------------------------------------
# Versión cuántica: Grover
# ---------------------------------------------------------------------

def construir_oraculo(n_qubits, estado_marcado):
    qc = QuantumCircuit(n_qubits)
    for i, bit in enumerate(reversed(estado_marcado)):
        if bit == "0":
            qc.x(i)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for i, bit in enumerate(reversed(estado_marcado)):
        if bit == "0":
            qc.x(i)
    return qc


def construir_difusor(n_qubits):
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def construir_circuito_grover(n_qubits, estado_marcado, iteraciones):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oraculo = construir_oraculo(n_qubits, estado_marcado)
    difusor = construir_difusor(n_qubits)
    for _ in range(iteraciones):
        qc.compose(oraculo, inplace=True)
        qc.compose(difusor, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def ejecutar_cuantico(n_qubits, device="CPU"):
    N = 2 ** n_qubits
    estado_marcado = format(random.randint(0, N - 1), f"0{n_qubits}b")
    iteraciones = max(1, math.floor((math.pi / 4) * math.sqrt(N)))

    simulador = AerSimulator(method="statevector", device=device)

    backend_real = None
    with ResourceMonitor() as mon:
        qc = construir_circuito_grover(n_qubits, estado_marcado, iteraciones)
        resultado = simulador.run(qc, shots=1024).result()
        resultado.get_counts()
        # Fuente de verdad: lo que Qiskit Aer reporta que usó realmente,
        # independiente de lo que se haya pedido con --device.
        try:
            backend_real = resultado.results[0].metadata.get("device")
        except Exception:
            backend_real = "desconocido"

    return mon.summary(), backend_real


# ---------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Grover vs búsqueda lineal")
    parser.add_argument("--min-qubits", type=int, default=2)
    parser.add_argument("--max-qubits", type=int, default=12)
    parser.add_argument("--device", choices=["CPU", "GPU"], default="CPU")
    parser.add_argument("--carpeta", default="csv", help="Carpeta donde se guarda el CSV (se crea si no existe)")
    parser.add_argument("--salida", default="resultados_grover.csv")
    args = parser.parse_args()

    os.makedirs(args.carpeta, exist_ok=True)
    ruta_salida = os.path.join(args.carpeta, args.salida)

    gpu_ok, gpu_err = gpu_monitoreo_disponible()
    if args.device == "GPU" and not gpu_ok:
        print(
            f"[AVISO] Pediste --device GPU pero el monitoreo de GPU (pynvml) "
            f"no se pudo inicializar: {gpu_err}. Los campos de GPU en el CSV "
            f"quedarán vacíos, pero Qiskit puede seguir usando la GPU igual; "
            f"revisa la columna 'grover_backend_real' para confirmarlo.",
            file=sys.stderr,
        )

    filas = []
    for n in range(args.min_qubits, args.max_qubits + 1):
        print(f"[Grover] n_qubits={n} (N={2 ** n})", file=sys.stderr)

        stats_clasico = ejecutar_clasico(n)
        stats_cuantico, backend_real = ejecutar_cuantico(n, device=args.device)

        print(f"    -> backend solicitado: {args.device} | backend real (Qiskit): {backend_real}", file=sys.stderr)

        fila = {"n_qubits": n, "N": 2 ** n}
        for k, v in stats_clasico.items():
            fila[f"clasico_{k}"] = v
        for k, v in stats_cuantico.items():
            fila[f"grover_{k}"] = v
        fila["grover_backend_solicitado"] = args.device
        fila["grover_backend_real"] = backend_real
        filas.append(fila)

    campos = list(filas[0].keys())
    with open(ruta_salida, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(filas)

    print(f"Resultados guardados en {ruta_salida}")


if __name__ == "__main__":
    main()
