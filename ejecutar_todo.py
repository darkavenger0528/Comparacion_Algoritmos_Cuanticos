"""
ejecutar_todo.py
----------------
Ejecuta TODOS los algoritmos (Grover, Deutsch-Jozsa, QFT y Shor), para
tamaños de 1 a 32 (o el rango que pidas), con varias repeticiones, para
ver cuánto varía cada medición entre la repetición 1, 2 y 3.

Cómo funciona (y por qué)
-------------------------
* NO modifica tus scripts: importa las funciones ejecutar_clasico() y
  ejecutar_cuantico() de grover.py, deutsch_jozsa.py, qft.py y shor.py.
* Cada medición corre en su PROPIO proceso (modo --worker). Así, si una
  corrida se queda sin memoria o se cuelga, solo muere esa corrida y no
  se pierde todo lo demás (con tus scripts originales, un fallo en n=28
  borra el CSV completo porque este se escribe al final).
* El lado clásico y el cuántico se miden por separado: si el simulador
  cuántico agota la memoria, el dato clásico del mismo tamaño se conserva.
* Un vigilante revisa la RAM del proceso: si pasa de --ram-max-pct del
  total del equipo, lo mata y lo registra como MEMORIA_EXCEDIDA, en vez de
  dejar que el equipo se congele por swap.
* Cada serie (algoritmo + lado + dispositivo) se detiene al primer fallo
  por recursos (memoria, tiempo límite o proceso matado). Ese punto es el
  "límite de viabilidad observado" y es un resultado, no un error.
  El tamaño donde falló sí se intenta en las 3 repeticiones.
* Antes de cada medición cuántica se hace una corrida de calentamiento
  (tamaño pequeño, no se registra) para no medir la inicialización de
  Qiskit/Aer/CUDA.
* Cada resultado se guarda en cuanto termina (csv/corridas_crudas.jsonl).
  Si se corta la luz o cierras la terminal, retomas con --reanudar.
* El orden de las series se baraja en cada repetición (semilla fija) para
  que el calentamiento del equipo no favorezca siempre a las mismas.

Uso
---
    # 1) Prueba rápida primero (unos minutos):
    python ejecutar_todo.py --max 10 --reps 1 --devices CPU

    # 2) Corrida completa pedida (1 a 32, 3 repeticiones, CPU y GPU):
    python ejecutar_todo.py --min 1 --max 32 --reps 3

    # 3) Si se interrumpe, retoma donde quedó:
    python ejecutar_todo.py --min 1 --max 32 --reps 3 --reanudar

    # 4) Solo regenerar los CSV a partir de lo ya medido:
    python ejecutar_todo.py --solo-resumen

Salidas (carpeta csv/)
----------------------
  corridas_crudas.jsonl    una línea por corrida (incluye fallos)
  todas_las_corridas.csv   lo mismo, en tabla (un dato por fila)
  resumen_variacion.csv    por condición: valor en rep1, rep2, rep3,
                           mediana y variación relativa entre repeticiones
  limites_viabilidad.csv   hasta qué tamaño llegó cada serie y por qué paró
  grover_cpu.csv, dj_cpu.csv, qft_cpu.csv, shor_cpu.csv (y *_gpu.csv)
                           formato que lee graficar_resultados.py, con la
                           MEDIANA de las repeticiones
"""

import argparse
import csv
import importlib
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time

import psutil

MARCA = "RESULTADO_JSON:"

# nombre en la línea de comandos -> (módulo, prefijo de columnas cuánticas, columna x)
ALGORITMOS = {
    "grover": {"modulo": "grover", "prefijo": "grover", "col_x": "n_qubits", "tiene_N": True},
    "dj":     {"modulo": "deutsch_jozsa", "prefijo": "dj", "col_x": "n_bits", "tiene_N": False},
    "qft":    {"modulo": "qft", "prefijo": "qft", "col_x": "n_qubits", "tiene_N": True},
    "shor":   {"modulo": "shor", "prefijo": "shor", "col_x": "parametro", "tiene_N": False},
}

METRICAS_NUMERICAS = [
    "tiempo_segundos", "cpu_promedio_pct", "cpu_maximo_pct",
    "ram_promedio_mb", "ram_maximo_mb", "ram_neto_mb",
    "gpu_uso_maximo_pct", "gpu_memoria_maxima_mb",
    "disco_lectura_mb", "disco_escritura_mb", "numero_muestras",
]

# Métricas que salen del muestreo periódico del monitor. Si la corrida duró tan poco
# que se tomaron menos de MIN_MUESTRAS muestras, esos valores no son confiables
# (el monitor reporta 0 aunque el proceso sí use memoria), así que se dejan vacíos.
MIN_MUESTRAS = 3
METRICAS_MONITOR = [m for m in METRICAS_NUMERICAS if m not in ("tiempo_segundos", "numero_muestras")]

PALABRAS_MEMORIA = ("memory", "memoria", "alloc", "insufficient", "out of")


# =====================================================================
#  MODO WORKER: una sola medición en un proceso aislado
# =====================================================================

def _llamar(mod, alg, lado, k, device):
    """Llama a la función del módulo y normaliza la salida a (stats, extra)."""
    if lado == "clasico":
        r = mod.ejecutar_clasico(k)
        if alg == "shor":
            stats, numero = r
            return stats, {"numero_factorizado": numero}
        return r, {}
    r = mod.ejecutar_cuantico(k, device=device)
    if alg == "shor":
        stats, factores, backend = r
        return stats, {"factores_encontrados": factores, "backend_real": backend}
    stats, backend = r
    return stats, {"backend_real": backend}


def worker(alg, lado, k, device, calentar):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    mod = importlib.import_module(ALGORITMOS[alg]["modulo"])

    if lado == "cuantico" and calentar:
        try:  # calentamiento: no se registra
            _llamar(mod, alg, lado, min(k, 3), device)
        except Exception:
            pass

    ram_base = psutil.Process().memory_info().rss / (1024 ** 2)
    stats, extra = _llamar(mod, alg, lado, k, device)

    stats = dict(stats)
    stats["ram_base_mb"] = round(ram_base, 2)
    # RAM "neta": lo que el proceso añadió sobre lo que ya ocupaba (Python + Qiskit)
    stats["ram_neto_mb"] = round(max(0.0, stats.get("ram_maximo_mb", 0) - ram_base), 2)
    stats.update(extra)
    print(MARCA + json.dumps(stats))


# =====================================================================
#  MODO ORQUESTADOR
# =====================================================================

def _es_fallo_de_recursos(rec):
    if rec["estado"] in ("TIMEOUT", "MEMORIA_EXCEDIDA", "PROCESO_MATADO"):
        return True
    msg = (rec.get("mensaje") or "").lower()
    return rec["estado"] == "ERROR" and any(p in msg for p in PALABRAS_MEMORIA)


def lanzar(alg, lado, k, device, rep, timeout, ram_max_mb, calentar):
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", alg, lado, str(k), device]
    if not calentar:
        cmd.append("--sin-calentamiento")

    rec = {"algoritmo": alg, "lado": lado, "dispositivo": device if lado == "cuantico" else "CPU",
           "k": k, "repeticion": rep, "estado": None, "mensaje": "", "stats": {}}

    with tempfile.TemporaryFile("w+") as fout, tempfile.TemporaryFile("w+") as ferr:
        t0 = time.time()
        p = subprocess.Popen(cmd, stdout=fout, stderr=ferr, text=True)
        ps = psutil.Process(p.pid)
        pico = 0.0
        estado = None
        while p.poll() is None:
            try:
                rss = ps.memory_info().rss / (1024 ** 2)
                pico = max(pico, rss)
                if rss > ram_max_mb:
                    p.kill()
                    estado = "MEMORIA_EXCEDIDA"
                    break
            except psutil.Error:
                pass
            if time.time() - t0 > timeout:
                p.kill()
                estado = "TIMEOUT"
                break
            time.sleep(0.1)
        p.wait()
        rec["tiempo_proceso_total_s"] = round(time.time() - t0, 3)
        rec["rss_pico_externo_mb"] = round(pico, 2)

        fout.seek(0)
        ferr.seek(0)
        salida = fout.read()
        error = ferr.read().strip()

    if estado:
        rec["estado"] = estado
        rec["mensaje"] = f"Cortado por el vigilante (límite {timeout}s / {ram_max_mb:.0f} MB)"
        return rec

    if p.returncode == 0:
        for linea in salida.splitlines():
            if linea.startswith(MARCA):
                rec["stats"] = json.loads(linea[len(MARCA):])
                rec["estado"] = "OK"
                return rec
        rec["estado"] = "ERROR"
        rec["mensaje"] = "El proceso terminó sin devolver resultado"
        return rec

    if p.returncode < 0:  # matado por una señal (típico: OOM-killer del sistema)
        rec["estado"] = "PROCESO_MATADO"
        rec["mensaje"] = f"Señal {-p.returncode} (probable falta de memoria del sistema)"
        return rec

    rec["estado"] = "ERROR"
    ultima = error.splitlines()[-1] if error else "sin mensaje"
    rec["mensaje"] = ultima[:300]
    return rec


def leer_jsonl(ruta):
    if not os.path.exists(ruta):
        return []
    with open(ruta, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def ejecutar_campana(args, ruta_jsonl):
    random.seed(args.semilla)
    ram_total_mb = psutil.virtual_memory().total / (1024 ** 2)
    ram_max_mb = ram_total_mb * args.ram_max_pct / 100

    devices = list(args.devices)
    if "GPU" in devices:
        from qiskit_aer import AerSimulator
        if "GPU" not in AerSimulator().available_devices():
            print("[AVISO] Aer no tiene GPU disponible aquí: se omite --devices GPU.", file=sys.stderr)
            devices.remove("GPU")

    previos = leer_jsonl(ruta_jsonl)
    hechos = {(r["algoritmo"], r["lado"], r["dispositivo"], r["k"], r["repeticion"]) for r in previos}
    limite = {}
    if args.reanudar:
        for r in previos:
            if r["estado"] != "OK" and _es_fallo_de_recursos(r):
                clave = (r["algoritmo"], r["lado"], r["dispositivo"])
                limite[clave] = min(limite.get(clave, 10 ** 9), r["k"])
    elif previos:
        print(f"[AVISO] Ya existe {ruta_jsonl} con {len(previos)} corridas. "
              f"Usa --reanudar para continuar o bórralo para empezar de cero.", file=sys.stderr)
        sys.exit(1)

    series = []
    for alg in args.algoritmos:
        series.append((alg, "clasico", "CPU"))
        for d in devices:
            series.append((alg, "cuantico", d))

    tamanos = list(range(args.min, args.max + 1))
    print(f"Equipo: {ram_total_mb / 1024:.1f} GB de RAM | vigilante de RAM en {ram_max_mb / 1024:.1f} GB "
          f"| tiempo límite por corrida: {args.timeout}s", file=sys.stderr)
    print(f"Series: {len(series)} | tamaños: {tamanos[0]}..{tamanos[-1]} | repeticiones: {args.reps}\n", file=sys.stderr)

    with open(ruta_jsonl, "a", encoding="utf-8") as fj:
        for rep in range(1, args.reps + 1):
            orden = series[:]
            random.shuffle(orden)  # el orden de las series cambia en cada repetición
            print(f"===== REPETICIÓN {rep}/{args.reps} =====", file=sys.stderr)
            for alg, lado, dev in orden:
                clave = (alg, lado, dev)
                consecutivos = 0
                for k in tamanos:
                    if k > limite.get(clave, 10 ** 9):
                        break
                    if (alg, lado, dev, k, rep) in hechos:
                        continue
                    rec = lanzar(alg, lado, k, dev, rep, args.timeout, ram_max_mb, not args.sin_calentamiento)
                    fj.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fj.flush()
                    hechos.add((alg, lado, dev, k, rep))

                    t = rec["stats"].get("tiempo_segundos")
                    detalle = f"{t}s" if t is not None else rec["mensaje"][:70]
                    print(f"[rep {rep}] {alg:7s} {lado:8s} {dev:3s} k={k:<2d} -> {rec['estado']:17s} {detalle}",
                          file=sys.stderr)

                    if rec["estado"] == "OK":
                        consecutivos = 0
                    elif _es_fallo_de_recursos(rec):
                        limite[clave] = min(limite.get(clave, 10 ** 9), k)
                        print(f"          >> límite de viabilidad de {alg}/{lado}/{dev}: k={k} "
                              f"({rec['estado']}). Se detiene esta serie.", file=sys.stderr)
                        break
                    else:
                        consecutivos += 1
                        if consecutivos >= 3:
                            print(f"          >> 3 errores seguidos en {alg}/{lado}/{dev}; se detiene esta serie.",
                                  file=sys.stderr)
                            break


# =====================================================================
#  GENERACIÓN DE CSV
# =====================================================================

def _limpiar(stats):
    """Devuelve una copia de stats con las métricas del monitor en None si hubo pocas muestras."""
    s2 = dict(stats)
    if s2.get("numero_muestras", 0) < MIN_MUESTRAS:
        for m in METRICAS_MONITOR:
            if m in s2:
                s2[m] = None
    return s2


def _mediana(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return statistics.median(vals) if vals else None


def generar_csv(carpeta, ruta_jsonl):
    registros = leer_jsonl(ruta_jsonl)
    if not registros:
        print("No hay corridas registradas todavía.", file=sys.stderr)
        return

    # ---------- 1) todas_las_corridas.csv ----------
    filas = []
    for r in registros:
        fila = {k: r[k] for k in ("algoritmo", "lado", "dispositivo", "k", "repeticion", "estado")}
        fila["mensaje"] = r.get("mensaje", "")
        fila["tiempo_proceso_total_s"] = r.get("tiempo_proceso_total_s")
        fila["rss_pico_externo_mb"] = r.get("rss_pico_externo_mb")
        fila.update(r.get("stats", {}))
        fila["monitor_confiable"] = r.get("stats", {}).get("numero_muestras", 0) >= MIN_MUESTRAS
        filas.append(fila)
    campos = []
    for f in filas:
        for c in f:
            if c not in campos:
                campos.append(c)
    with open(os.path.join(carpeta, "todas_las_corridas.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)

    # agrupar corridas exitosas por condición
    ok = [r for r in registros if r["estado"] == "OK"]
    cond = {}
    for r in ok:
        cond.setdefault((r["algoritmo"], r["lado"], r["dispositivo"], r["k"]), []).append(r)

    # ---------- 2) resumen_variacion.csv ----------
    maximo_reps = max((r["repeticion"] for r in registros), default=1)
    cols = ["algoritmo", "lado", "dispositivo", "k", "metrica"] + \
           [f"rep{i}" for i in range(1, maximo_reps + 1)] + ["mediana", "variacion_pct"]
    with open(os.path.join(carpeta, "resumen_variacion.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for (alg, lado, dev, k), lista in sorted(cond.items()):
            for m in ("tiempo_segundos", "ram_maximo_mb", "ram_neto_mb", "cpu_promedio_pct",
                      "gpu_memoria_maxima_mb", "disco_lectura_mb", "disco_escritura_mb"):
                por_rep = {r["repeticion"]: _limpiar(r["stats"]).get(m) for r in lista}
                vals = [v for v in por_rep.values() if isinstance(v, (int, float))]
                if not vals:
                    continue
                med = statistics.median(vals)
                var = round((max(vals) - min(vals)) / med * 100, 2) if len(vals) > 1 and med else None
                fila = {"algoritmo": alg, "lado": lado, "dispositivo": dev, "k": k, "metrica": m,
                        "mediana": round(med, 4), "variacion_pct": var}
                for i in range(1, maximo_reps + 1):
                    fila[f"rep{i}"] = por_rep.get(i)
                w.writerow(fila)

    # ---------- 3) limites_viabilidad.csv ----------
    series = {}
    for r in registros:
        series.setdefault((r["algoritmo"], r["lado"], r["dispositivo"]), []).append(r)
    with open(os.path.join(carpeta, "limites_viabilidad.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["algoritmo", "lado", "dispositivo", "k_max_exitoso_en_todas_las_reps",
                    "primer_fallo_k", "motivo_primer_fallo"])
        for (alg, lado, dev), lista in sorted(series.items()):
            reps = {r["repeticion"] for r in lista}
            por_k_ok = {}
            for r in lista:
                if r["estado"] == "OK":
                    por_k_ok.setdefault(r["k"], set()).add(r["repeticion"])
            k_ok = max((k for k, s in por_k_ok.items() if s == reps), default=None)
            fallos = sorted((r["k"], r["estado"], r.get("mensaje", "")) for r in lista if r["estado"] != "OK")
            if fallos:
                w.writerow([alg, lado, dev, k_ok, fallos[0][0], f"{fallos[0][1]}: {fallos[0][2]}"[:200]])
            else:
                w.writerow([alg, lado, dev, k_ok, "", "sin fallos"])

    # ---------- 4) CSV compatibles con graficar_resultados.py (medianas) ----------
    for alg, info in ALGORITMOS.items():
        for dev in ("CPU", "GPU"):
            claves = {(a, l, d, k) for (a, l, d, k) in cond if a == alg}
            # Mismas filas en el archivo CPU y en el GPU (con huecos donde falte dato):
            # graficar_resultados.py usa el eje X del CSV de CPU para dibujar también la GPU.
            ks = sorted({k for (a, l, d, k) in claves})
            if not any(l == "cuantico" and d == dev for (a, l, d, k) in claves):
                continue

            def agregar(lado, d, k):
                lista = cond.get((alg, lado, d, k), [])
                if not lista:
                    return {}
                limpias = [_limpiar(r["stats"]) for r in lista]
                claves_stats = {c for st in limpias for c in st}
                salida = {}
                for c in claves_stats:
                    vals = [st.get(c) for st in limpias]
                    if any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
                        salida[c] = _mediana(vals)
                    else:
                        salida[c] = next((v for v in vals if v is not None), None)
                return salida

            ruta = os.path.join(carpeta, f"{alg}_{dev.lower()}.csv")
            if alg != "shor":
                filas_w = []
                for k in ks:
                    fila = {info["col_x"]: k}
                    if info["tiene_N"]:
                        fila["N"] = 2 ** k
                    for c, v in agregar("clasico", "CPU", k).items():
                        fila[f"clasico_{c}"] = v
                    for c, v in agregar("cuantico", dev, k).items():
                        fila[f"{info['prefijo']}_{c}"] = v
                    filas_w.append(fila)
            else:  # Shor usa formato largo con columna 'tipo'
                filas_w = []
                for k in ks:
                    c_ = agregar("clasico", "CPU", k)
                    if c_:
                        filas_w.append({"tipo": "clasico", "parametro": k, **c_})
                    q_ = agregar("cuantico", dev, k)
                    if q_:
                        filas_w.append({"tipo": "shor", "parametro": k, **q_})
            if filas_w:
                campos_w = []
                for fila in filas_w:
                    for c in fila:
                        if c not in campos_w:
                            campos_w.append(c)
                with open(ruta, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=campos_w)
                    w.writeheader()
                    w.writerows(filas_w)

    print(f"\nCSV generados en {carpeta}/ (todas_las_corridas, resumen_variacion, "
          f"limites_viabilidad y *_cpu/_gpu para graficar_resultados.py)", file=sys.stderr)


# =====================================================================

def main():
    if len(sys.argv) >= 6 and sys.argv[1] == "--worker":
        calentar = "--sin-calentamiento" not in sys.argv
        worker(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], calentar)
        return

    ap = argparse.ArgumentParser(description="Corre todos los algoritmos, tamaños y repeticiones.")
    ap.add_argument("--min", type=int, default=1, help="tamaño mínimo (qubits / bits)")
    ap.add_argument("--max", type=int, default=32, help="tamaño máximo (qubits / bits)")
    ap.add_argument("--reps", type=int, default=3, help="repeticiones por condición")
    ap.add_argument("--algoritmos", nargs="+", choices=list(ALGORITMOS), default=list(ALGORITMOS))
    ap.add_argument("--devices", nargs="+", choices=["CPU", "GPU"], default=["CPU", "GPU"])
    ap.add_argument("--timeout", type=int, default=600, help="segundos máximos por corrida")
    ap.add_argument("--ram-max-pct", type=float, default=85.0,
                    help="si un proceso pasa de este %% de la RAM total, se mata y se registra")
    ap.add_argument("--semilla", type=int, default=12345)
    ap.add_argument("--carpeta", default="csv")
    ap.add_argument("--reanudar", action="store_true")
    ap.add_argument("--sin-calentamiento", action="store_true")
    ap.add_argument("--solo-resumen", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.carpeta, exist_ok=True)
    ruta_jsonl = os.path.join(args.carpeta, "corridas_crudas.jsonl")

    if not args.solo_resumen:
        ejecutar_campana(args, ruta_jsonl)
    generar_csv(args.carpeta, ruta_jsonl)


if __name__ == "__main__":
    main()
