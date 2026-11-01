"""
ibm_grover.py
-------------
Corre el algoritmo de Grover en un procesador cuántico REAL de IBM
(Open Plan, gratis) y lo compara contra la simulación local ideal
(sin ruido) del mismo circuito.

IMPORTANTE — esto NO mide consumo de recursos de tu computador:
a diferencia de grover.py, aquí el cálculo cuántico ocurre físicamente
en un chip de IBM, no en tu laptop. Lo que se mide aquí es otra cosa,
igual de válida para el proyecto pero distinta:
  - tiempo real de cómputo cuántico (segundos de QPU consumidos)
  - qué tan bien funciona el algoritmo en hardware real con ruido,
    comparado contra el resultado ideal (sin ruido) de la simulación

Esto sirve para una tercera comparación en el proyecto: simulación
local (Aer, CPU/GPU) vs. hardware cuántico real (IBM), en vez de
simulación vs. clásico.

ADVERTENCIA SOBRE EL PRESUPUESTO GRATUITO:
El Open Plan de IBM da unos 10 minutos de tiempo de QPU gratis por mes.
Cada ejecución de este script consume parte de ese presupuesto. EMPIEZA
con --min-qubits 2 --max-qubits 2 para probar que todo funciona antes
de correr un rango más amplio. Puedes revisar cuánto te queda en el
dashboard de quantum.cloud.ibm.com.

Requiere haber guardado tus credenciales antes (ver instrucciones):
    from qiskit_ibm_runtime import QiskitRuntimeService
    QiskitRuntimeService.save_account(
        channel="ibm_quantum_platform", token="TU_API_KEY",
        instance="TU_CRN", set_as_default=True,
    )
"""

import argparse
import csv
import math
import os
import random
import sys

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2


# ---------------------------------------------------------------------
# Construcción del circuito de Grover (igual que en grover.py)
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


# ---------------------------------------------------------------------
# Simulación local ideal (referencia, sin ruido)
# ---------------------------------------------------------------------

def tasa_exito_ideal(qc, estado_marcado, shots=1024):
    simulador = AerSimulator(method="statevector", device="CPU")
    resultado = simulador.run(qc, shots=shots).result()
    conteos = resultado.get_counts()
    total = sum(conteos.values())
    exitos = conteos.get(estado_marcado, 0)
    return exitos / total


# ---------------------------------------------------------------------
# Ejecución en hardware real de IBM
# ---------------------------------------------------------------------

def ejecutar_en_ibm(service, n_qubits, estado_marcado, iteraciones, shots):
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=n_qubits)
    print(f"    -> backend real de IBM elegido: {backend.name} ({backend.num_qubits} qubits)", file=sys.stderr)

    qc = construir_circuito_grover(n_qubits, estado_marcado, iteraciones)

    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    qc_isa = pm.run(qc)

    sampler = SamplerV2(mode=backend)
    print("    -> enviando job (puede tardar en cola, eso NO consume tu presupuesto gratis)...", file=sys.stderr)
    job = sampler.run([qc_isa], shots=shots)
    print(f"    -> job_id: {job.job_id()} (puedes revisarlo en quantum.cloud.ibm.com)", file=sys.stderr)

    resultado = job.result()
    conteos = resultado[0].data.c.get_counts()
    total = sum(conteos.values())
    exitos = conteos.get(estado_marcado, 0)
    tasa_exito_real = exitos / total

    tiempo_qpu = None
    try:
        tiempo_qpu = job.usage()  # segundos de QPU consumidos (cuenta contra tu presupuesto)
    except Exception as e:
        print(f"    -> [aviso] no se pudo leer job.usage(): {e}", file=sys.stderr)

    return {
        "backend_ibm": backend.name,
        "job_id": job.job_id(),
        "tiempo_qpu_segundos": tiempo_qpu,
        "tasa_exito_medida": round(tasa_exito_real, 4),
    }


# ---------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Grover en hardware cuántico real de IBM vs. simulación local ideal"
    )
    parser.add_argument("--min-qubits", type=int, default=2)
    parser.add_argument("--max-qubits", type=int, default=2,
                         help="EMPIEZA EN 2. Cada qubit adicional consume más de tu presupuesto gratis.")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--carpeta", default="csv")
    parser.add_argument("--salida", default="grover_ibm_real.csv")
    args = parser.parse_args()

    print("Conectando con IBM Quantum...", file=sys.stderr)
    service = QiskitRuntimeService()

    os.makedirs(args.carpeta, exist_ok=True)
    ruta_salida = os.path.join(args.carpeta, args.salida)

    filas = []
    for n in range(args.min_qubits, args.max_qubits + 1):
        print(f"[Grover-IBM] n_qubits={n}", file=sys.stderr)

        N = 2 ** n
        estado_marcado = format(random.randint(0, N - 1), f"0{n}b")
        iteraciones = max(1, math.floor((math.pi / 4) * math.sqrt(N)))

        qc_local = construir_circuito_grover(n, estado_marcado, iteraciones)
        exito_ideal = tasa_exito_ideal(qc_local, estado_marcado, shots=args.shots)

        datos_ibm = ejecutar_en_ibm(service, n, estado_marcado, iteraciones, args.shots)

        fila = {
            "n_qubits": n,
            "estado_marcado": estado_marcado,
            "tasa_exito_teorica_ideal": round(exito_ideal, 4),
        }
        fila.update(datos_ibm)
        filas.append(fila)

        print(
            f"    -> éxito ideal (simulado, sin ruido): {exito_ideal:.2%} | "
            f"éxito real (hardware IBM, con ruido): {datos_ibm['tasa_exito_medida']:.2%}",
            file=sys.stderr,
        )

    campos = list(filas[0].keys())
    with open(ruta_salida, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(filas)

    print(f"\nResultados guardados en {ruta_salida}")
    print("Revisa tu consumo de minutos gratis en https://quantum.cloud.ibm.com")


if __name__ == "__main__":
    main()
