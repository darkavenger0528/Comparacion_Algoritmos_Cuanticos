"""
verificaciones.py
-----------------
Dos comprobaciones complementarias que se citan en el documento
(secciones 8.3 y 9.1). Ejecútalas en TU equipo, con tu venv, junto a
grover.py y deutsch_jozsa.py:

    venv/bin/python verificaciones.py

1) Método que Aer usa en Deutsch-Jozsa, en CPU y en GPU.
   Si dice 'stabilizer', el circuito NO se simula con vector de estado
   (por eso el tiempo y la RAM no crecen). Revisa también el campo
   'device' para ver si la GPU participó de verdad.

2) Dónde aparece el exceso de RAM de Grover: el pico durante la
   ejecución en Aer frente al tamaño del vector 16*2^n. Si el exceso crece
   con el número de instrucciones del circuito y no con 2^n, se confirma
   lo que dice la sección 8.3. (Tarda 1 o 2 minutos.)
"""

import math
import resource
import time

import psutil
from qiskit_aer import AerSimulator

import deutsch_jozsa as dj
import grover


def mb_pico():
    # ru_maxrss está en KiB en Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def verificar_metodo_dj(n=10):
    print("=== 1) Método que usa Aer en Deutsch-Jozsa ===")
    disponibles = AerSimulator().available_devices()
    print("Dispositivos disponibles en Aer:", disponibles)
    for dev in ("CPU", "GPU"):
        if dev not in disponibles:
            print(f"  {dev}: no disponible, se omite")
            continue
        qc = dj.construir_circuito_dj(n, dj.generar_secreto(n))
        meta = AerSimulator(device=dev).run(qc, shots=64).result().results[0].metadata
        print(f"  Pedido: {dev} | method = {meta.get('method')} | device = {meta.get('device')}")
    qc = dj.construir_circuito_dj(n, dj.generar_secreto(n))
    meta = AerSimulator(method="statevector").run(qc, shots=64).result().results[0].metadata
    print(f"  Forzando statevector: method = {meta.get('method')}")
    print()


def verificar_ram_grover(tamanos=(17, 18, 19)):
    print("=== 2) Exceso de RAM de Grover durante run() de Aer (CPU) ===")
    proc = psutil.Process()
    print(f"{'n':>3} {'iteraciones':>12} {'instrucciones':>14} {'RAM antes (MB)':>15} {'pico (MB)':>10} {'exceso (MB)':>12} {'16*2^n (MB)':>12} {'t (s)':>7}")
    for n in tamanos:
        iteraciones = max(1, math.floor((math.pi / 4) * math.sqrt(2 ** n)))
        qc = grover.construir_circuito_grover(n, format(1, f"0{n}b"), iteraciones)
        sim = AerSimulator(method="statevector", device="CPU")
        antes = proc.memory_info().rss / 2 ** 20
        t0 = time.time()
        sim.run(qc, shots=1024).result().get_counts()
        dt = time.time() - t0
        pico = mb_pico()
        print(f"{n:>3} {iteraciones:>12} {len(qc.data):>14} {antes:>15.0f} {pico:>10.0f} {pico - antes:>12.0f} {16 * 2 ** n / 2 ** 20:>12.0f} {dt:>7.1f}")
    print("\nSi 'exceso' sube con las instrucciones y es mucho mayor que 16*2^n, el exceso")
    print("depende del largo del circuito y no del vector de estado.")


if __name__ == "__main__":
    verificar_metodo_dj()
    verificar_ram_grover()
