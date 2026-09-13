"""
shor.py
-------
Compara la factorización clásica por división de prueba contra el
algoritmo de Shor simulado en Qiskit.

IMPORTANTE — limitación práctica documentada:
Simular Shor para números grandes requiere un número de qubits que
crece con log2(N), y el costo de la simulación crece exponencialmente
con esos qubits. Por eso aquí la parte cuántica se limita al caso de
demostración estándar N = 15 (factorizado como 3 x 5), variando en su
lugar el número de qubits del registro de conteo (precisión de la
estimación de fase), que es lo que realmente controla el tamaño y
costo del circuito simulado.

Como la parte clásica y la cuántica no comparten el mismo eje de
"tamaño de problema" (una escala en bits del número a factorizar, la
otra en qubits de precisión), este script genera DOS tablas separadas
dentro del mismo CSV, identificadas por la columna `tipo`, en lugar de
forzarlas a un único formato ancho fila-por-fila.

La columna `shor_backend_real` (solo en las filas tipo='shor') es lo
que Qiskit Aer reporta que usó de verdad (CPU o GPU) en sus metadatos.
"""

import argparse
import csv
import math
import os
import sys
from fractions import Fraction

from resource_monitor import ResourceMonitor, gpu_monitoreo_disponible

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------
# Versión clásica: división de prueba
# ---------------------------------------------------------------------

def factorizar_por_division(n):
    factores = []
    d = 2
    resto = n
    while d * d <= resto:
        while resto % d == 0:
            factores.append(d)
            resto //= d
        d += 1
    if resto > 1:
        factores.append(resto)
    return factores


def ejecutar_clasico(n_bits_numero):
    from sympy import nextprime

    base = 2 ** (n_bits_numero // 2)
    p = nextprime(base)
    q = nextprime(p + 1)
    numero = p * q

    with ResourceMonitor() as mon:
        factorizar_por_division(numero)
    return mon.summary(), numero


# ---------------------------------------------------------------------
# Versión cuántica: Shor (N = 15)
# ---------------------------------------------------------------------

def controlado_a_mod15(a, potencia):
    """
    Compuerta controlada que multiplica por a^potencia mod 15.

    Optimización importante: el orden multiplicativo de cualquier base
    válida módulo 15 divide a 4 (a^4 ≡ 1 mod 15 siempre). Por lo tanto,
    en vez de repetir literalmente el patrón de swaps 'potencia' veces
    (lo cual crece exponencialmente para los qubits de conteo más
    significativos y no aporta nada, porque matemáticamente se cancela
    a la identidad), reducimos el exponente módulo 4 antes de construir
    el circuito. Esto mantiene el circuito con profundidad O(1) por
    controlada, en vez de O(2^q), reflejando el costo real de simular
    más qubits en lugar de un artefacto de una construcción ineficiente.
    """
    if a not in [2, 4, 7, 8, 11, 13]:
        raise ValueError("a debe ser 2, 4, 7, 8, 11 o 13 para N=15")

    pasos_efectivos = potencia % 4  # el orden multiplicativo siempre divide a 4

    U = QuantumCircuit(4)
    for _ in range(pasos_efectivos):
        if a in [2, 13]:
            U.swap(2, 3)
            U.swap(1, 2)
            U.swap(0, 1)
        if a in [7, 8]:
            U.swap(0, 1)
            U.swap(1, 2)
            U.swap(2, 3)
        if a in [4, 11]:
            U.swap(1, 3)
            U.swap(0, 2)
        if a in [7, 11, 13]:
            for q in range(4):
                U.x(q)

    gate = U.to_gate()
    gate.name = f"{a}^{potencia} mod 15"
    return gate.control()


def construir_circuito_shor(n_count, a=7):
    qc = QuantumCircuit(n_count + 4, n_count)
    qc.x(n_count)
    qc.h(range(n_count))

    for q in range(n_count):
        qc.append(controlado_a_mod15(a, 2 ** q), [q] + list(range(n_count, n_count + 4)))

    qc.append(QFT(n_count, inverse=True), range(n_count))
    qc.measure(range(n_count), range(n_count))
    return qc


def interpretar_resultado(conteos, n_count):
    medicion_mas_comun = max(conteos, key=conteos.get)
    fase = int(medicion_mas_comun, 2) / (2 ** n_count)
    fraccion = Fraction(fase).limit_denominator(15)
    return fraccion.denominator


def ejecutar_cuantico(n_count, device="CPU", a=7, N=15):
    simulador = AerSimulator(device=device)

    backend_real = None
    with ResourceMonitor() as mon:
        qc = construir_circuito_shor(n_count, a=a)
        qc = transpile(qc, simulador)
        resultado = simulador.run(qc, shots=2048).result()
        conteos = resultado.get_counts()
        r = interpretar_resultado(conteos, n_count)
        factores_encontrados = None
        if r % 2 == 0:
            candidato1 = math.gcd(a ** (r // 2) - 1, N)
            candidato2 = math.gcd(a ** (r // 2) + 1, N)
            factores_encontrados = f"{candidato1},{candidato2}"
        try:
            backend_real = resultado.results[0].metadata.get("device")
        except Exception:
            backend_real = "desconocido"

    return mon.summary(), factores_encontrados, backend_real


# ---------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Shor (N=15, precisión variable) vs división de prueba clásica"
    )
    parser.add_argument("--min-count-qubits", type=int, default=3)
    parser.add_argument("--max-count-qubits", type=int, default=10)
    parser.add_argument("--min-bits-clasico", type=int, default=4)
    parser.add_argument("--max-bits-clasico", type=int, default=40)
    parser.add_argument("--paso-clasico", type=int, default=4)
    parser.add_argument("--device", choices=["CPU", "GPU"], default="CPU")
    parser.add_argument("--carpeta", default="csv", help="Carpeta donde se guarda el CSV (se crea si no existe)")
    parser.add_argument("--salida", default="resultados_shor.csv")
    args = parser.parse_args()

    os.makedirs(args.carpeta, exist_ok=True)
    ruta_salida = os.path.join(args.carpeta, args.salida)

    gpu_ok, gpu_err = gpu_monitoreo_disponible()
    if args.device == "GPU" and not gpu_ok:
        print(
            f"[AVISO] Pediste --device GPU pero el monitoreo de GPU (pynvml) "
            f"no se pudo inicializar: {gpu_err}. Revisa 'backend_real' para "
            f"confirmar qué usó Qiskit realmente.",
            file=sys.stderr,
        )

    filas = []

    print("== Parte clásica: división de prueba sobre números crecientes ==", file=sys.stderr)
    for n_bits in range(args.min_bits_clasico, args.max_bits_clasico + 1, args.paso_clasico):
        print(f"[Clásico] ~{n_bits} bits", file=sys.stderr)
        stats, numero = ejecutar_clasico(n_bits)
        fila = {"tipo": "clasico", "parametro": n_bits, "numero_factorizado": numero}
        fila.update(stats)
        filas.append(fila)

    print("== Parte cuántica: Shor sobre N=15 variando qubits de conteo ==", file=sys.stderr)
    for n_count in range(args.min_count_qubits, args.max_count_qubits + 1):
        print(f"[Shor] n_count={n_count}", file=sys.stderr)
        stats, factores, backend_real = ejecutar_cuantico(n_count, device=args.device)
        print(f"    -> backend solicitado: {args.device} | backend real (Qiskit): {backend_real}", file=sys.stderr)
        fila = {
            "tipo": "shor",
            "parametro": n_count,
            "factores_encontrados": factores,
            "backend_solicitado": args.device,
            "backend_real": backend_real,
        }
        fila.update(stats)
        filas.append(fila)

    campos = sorted({k for fila in filas for k in fila.keys()})
    with open(ruta_salida, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(filas)

    print(f"Resultados guardados en {ruta_salida}")


if __name__ == "__main__":
    main()
