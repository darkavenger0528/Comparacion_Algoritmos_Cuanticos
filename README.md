# Proyecto: Algoritmos clásicos vs. cuánticos simulados en Qiskit

Este directorio contiene los 4 algoritmos comparados en el proyecto. Cada
script ejecuta la versión clásica y la versión cuántica para tamaños de
problema crecientes, midiendo RAM, CPU, disco y GPU (si está disponible)
en cada ejecución, y guardando todo en un CSV **en formato ancho**: una
fila por tamaño de problema, con columnas separadas por prefijo
(`clasico_...`, `grover_...`, etc.), para que sea fácil de leer y de
graficar directamente sin tener que filtrar filas por algoritmo.

## Archivos

- `resource_monitor.py` — módulo compartido que mide recursos (no se
  ejecuta directamente).
- `grover.py` — búsqueda lineal clásica vs. algoritmo de Grover.
- `deutsch_jozsa.py` — evaluación por fuerza bruta vs. Deutsch-Jozsa.
- `qft.py` — FFT clásica (NumPy) vs. Transformada Cuántica de Fourier.
- `shor.py` — división de prueba clásica vs. algoritmo de Shor (N=15).

## Cómo ejecutar

```fish
venv/bin/python grover.py --min-qubits 2 --max-qubits 16 --device GPU --salida grover_gpu.csv
venv/bin/python grover.py --min-qubits 2 --max-qubits 16 --device CPU --salida grover_cpu.csv
```

(y lo mismo para `deutsch_jozsa.py`, `qft.py` y `shor.py`)

## ¿Cómo saber si de verdad se usó la GPU?

**No confíes en herramientas externas como btop para esto** — la
numeración de GPU0/GPU1 que muestra no necesariamente corresponde a
"integrada" vs. "dedicada" en el orden que uno esperaría, y es fácil
malinterpretarla.

La fuente de verdad correcta es Qiskit mismo: cada fila del CSV que
corresponde a la parte cuántica incluye dos columnas:

- `..._backend_solicitado`: lo que le pediste al script con `--device`.
- `..._backend_real`: lo que **Qiskit Aer reporta en los metadatos del
  resultado** que usó de verdad para simular ese circuito.

Si pediste `GPU` y `..._backend_real` dice `CPU`, significa que Aer no
pudo usar la GPU para ese caso (por ejemplo, algunos métodos de
simulación o tamaños de circuito muy pequeños a veces no activan el
backend GPU) y usó CPU como respaldo silencioso. Esto es lo que hay que
revisar, no el panel de btop.

Además, cada fila trae `..._gpu_monitoreo_activo` (True/False) y
`..._gpu_monitoreo_error`: si ves `False` y un mensaje de error ahí, es
que la librería `pynvml`/`nvidia-ml-py` no pudo inicializarse en tu
máquina para *medir* la GPU (uso %, memoria), aunque Qiskit sí la haya
usado para calcular. Son dos cosas independientes: una es "¿Qiskit usó
la GPU?" (columna `backend_real`) y otra es "¿pudimos medir cuánto la
usó?" (columnas `gpu_uso_maximo_pct`, `gpu_memoria_maxima_mb`).

Si necesitas instalar la librería de monitoreo de GPU:

```fish
uv pip install --python venv/bin/python nvidia-ml-py
```

(reemplaza al antiguo paquete `pynvml`, que está deprecado, pero se
importa exactamente igual, así que no hace falta cambiar nada del
código).

## Columnas del CSV

Para cada rama (`clasico_`, `grover_`/`dj_`/`qft_`/`shor_`):

- `tiempo_segundos`: tiempo total de ejecución.
- `cpu_promedio_pct` / `cpu_maximo_pct`: uso de CPU del proceso durante
  la ejecución (puede superar 100% en máquinas multi-núcleo: 400%
  significa el equivalente a 4 núcleos completos ocupados).
- `ram_promedio_mb` / `ram_maximo_mb`: memoria RAM usada por el proceso.
- `gpu_uso_maximo_pct` / `gpu_memoria_maxima_mb`: uso de GPU medido con
  pynvml/nvidia-ml-py (vacío si no se pudo medir; ver sección anterior).
- `disco_lectura_mb` / `disco_escritura_mb`: lectura/escritura a disco.
- `numero_muestras`: cuántas veces se muestreó (útil para saber si la
  ejecución fue tan rápida que apenas se tomó 1 muestra).

## Notas importantes por algoritmo

- **Grover y QFT** escalan de forma directa con el número de qubits
  (`N = 2^n_qubits`). Cuidado al subir `--max-qubits` por encima de
  ~20-22: la memoria requerida para el vector de estado crece como
  2^n × 16 bytes; a partir de cierto punto te puedes quedar sin RAM o
  VRAM. Sube el límite gradualmente y observa el consumo.
- **Deutsch-Jozsa** es más liviano (el oráculo no usa compuertas
  multi-control profundas), así que puedes llegar a más qubits antes
  de tocar el mismo límite.
- **Shor** está limitado a N=15 (caso estándar de demostración); la
  parte cuántica escala el número de qubits del *registro de conteo*
  en vez del tamaño del número a factorizar, mientras que la parte
  clásica sí escala libremente. El CSV de Shor tiene una estructura
  distinta a los otros tres (columna `tipo` con valores `clasico`/
  `shor` en vez de formato ancho), precisamente porque estos dos ejes
  de "tamaño de problema" no son comparables directamente. Esta
  asimetría es intencional y vale la pena discutirla en el informe.
- El algoritmo de Shor es probabilístico: no todas las mediciones
  producen factores no triviales de 15 (esperado según la teoría). Se
  reporta lo encontrado en cada corrida en `factores_encontrados`.

## Análisis de resultados

Con los CSV generados (idealmente una corrida en CPU y otra en GPU por
algoritmo), el siguiente paso es graficar `tiempo_segundos`,
`ram_maximo_mb` y `gpu_memoria_maxima_mb` en función del tamaño del
problema, para cada algoritmo y cada backend (objetivo específico 5 del
proyecto).
