"""
resource_monitor.py
--------------------
Utilidad para medir el consumo de recursos (RAM, CPU, disco y, si está
disponible, GPU NVIDIA) durante la ejecución de un bloque de código.

Uso típico:

    from resource_monitor import ResourceMonitor

    with ResourceMonitor() as mon:
        hacer_algo_costoso()

    print(mon.summary())

El monitoreo se hace en un hilo aparte que toma muestras cada `interval`
segundos mientras el bloque `with` está en ejecución. Al salir del bloque,
`summary()` devuelve un diccionario con los valores promedio y máximos
observados, más el tiempo total transcurrido.

Si la librería `pynvml` no está instalada o no hay una GPU NVIDIA
disponible, las métricas de GPU simplemente se reportan como None, sin
que el resto del monitoreo se vea afectado.
"""

import os
import time
import threading

import psutil

_GPU_DISPONIBLE = False
_GPU_ERROR = None
try:
    import pynvml
    pynvml.nvmlInit()
    _GPU_DISPONIBLE = True
except Exception as e:
    _GPU_ERROR = str(e)


def gpu_monitoreo_disponible():
    """Indica si pynvml pudo inicializarse, y por qué si no."""
    return _GPU_DISPONIBLE, _GPU_ERROR


class ResourceMonitor:
    def __init__(self, interval=0.05, gpu_index=0):
        self.interval = interval
        self.gpu_index = gpu_index

        self._process = psutil.Process(os.getpid())
        self._samples = []
        self._stop_event = threading.Event()
        self._thread = None

        self._disk_before = None
        self._disk_after = None
        self._t0 = None
        self._t1 = None

        self._gpu_handle = None
        self.gpu_error = _GPU_ERROR
        if _GPU_DISPONIBLE:
            try:
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            except Exception as e:
                self.gpu_error = str(e)

    def _tomar_muestra(self):
        # OJO: se usa la variante BLOQUEANTE de cpu_percent (interval=self.interval)
        # en lugar de interval=None. Llamar a cpu_percent(interval=None) en un bucle
        # muy rápido produce mediciones erráticas (a veces >1000%) porque psutil
        # calcula el porcentaje dividiendo el tiempo de CPU usado entre el tiempo de
        # reloj transcurrido desde la última llamada, y ese intervalo real puede ser
        # mucho más corto de lo esperado por la forma en que el hilo es planificado.
        # Bloquear durante 'interval' segundos garantiza una ventana de medición
        # exacta y por lo tanto un porcentaje confiable.
        cpu_percent = self._process.cpu_percent(interval=self.interval)
        mem_info = self._process.memory_info()
        ram_mb = mem_info.rss / (1024 ** 2)

        gpu_util = None
        gpu_mem_mb = None
        if self._gpu_handle is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                gpu_util = util.gpu
                gpu_mem_mb = mem.used / (1024 ** 2)
            except Exception:
                pass

        return {
            "cpu_percent": cpu_percent,
            "ram_mb": ram_mb,
            "gpu_util_percent": gpu_util,
            "gpu_mem_mb": gpu_mem_mb,
        }

    def _ciclo_muestreo(self):
        # Llamada de "calentamiento": la primera lectura de cpu_percent no tiene
        # una ventana previa con la que compararse y psutil la descarta igual,
        # pero la hacemos explícita para dejar el cronómetro interno en marcha.
        self._process.cpu_percent(interval=None)
        while not self._stop_event.is_set():
            # _tomar_muestra() ya bloquea internamente 'interval' segundos
            # (ver nota en _tomar_muestra), así que no hace falta time.sleep aquí.
            self._samples.append(self._tomar_muestra())

    def __enter__(self):
        self._disk_before = psutil.disk_io_counters()
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._ciclo_muestreo, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._t1 = time.time()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._disk_after = psutil.disk_io_counters()
        return False  # no suprime excepciones

    @property
    def elapsed_seconds(self):
        return (self._t1 or time.time()) - self._t0

    def summary(self):
        if self._samples:
            cpu_vals = [s["cpu_percent"] for s in self._samples]
            ram_vals = [s["ram_mb"] for s in self._samples]
            gpu_util_vals = [s["gpu_util_percent"] for s in self._samples if s["gpu_util_percent"] is not None]
            gpu_mem_vals = [s["gpu_mem_mb"] for s in self._samples if s["gpu_mem_mb"] is not None]
        else:
            cpu_vals, ram_vals, gpu_util_vals, gpu_mem_vals = [0], [0], [], []

        disk_read_mb = disk_write_mb = None
        if self._disk_before is not None and self._disk_after is not None:
            disk_read_mb = (self._disk_after.read_bytes - self._disk_before.read_bytes) / (1024 ** 2)
            disk_write_mb = (self._disk_after.write_bytes - self._disk_before.write_bytes) / (1024 ** 2)

        return {
            "tiempo_segundos": round(self.elapsed_seconds, 4),
            "cpu_promedio_pct": round(sum(cpu_vals) / len(cpu_vals), 2),
            "cpu_maximo_pct": round(max(cpu_vals), 2),
            "ram_promedio_mb": round(sum(ram_vals) / len(ram_vals), 2),
            "ram_maximo_mb": round(max(ram_vals), 2),
            "gpu_uso_maximo_pct": round(max(gpu_util_vals), 2) if gpu_util_vals else None,
            "gpu_memoria_maxima_mb": round(max(gpu_mem_vals), 2) if gpu_mem_vals else None,
            "disco_lectura_mb": round(disk_read_mb, 4) if disk_read_mb is not None else None,
            "disco_escritura_mb": round(disk_write_mb, 4) if disk_write_mb is not None else None,
            "numero_muestras": len(self._samples),
            "gpu_monitoreo_activo": self._gpu_handle is not None,
            "gpu_monitoreo_error": self.gpu_error,
        }
