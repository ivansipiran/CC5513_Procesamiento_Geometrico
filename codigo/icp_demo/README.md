# Demo interactivo de ICP (Iterative Closest Point)

Herramienta de consola para la clase de registro de nubes de puntos. Ejecuta el
ICP clasico sobre datos reales (Stanford Bunny, Armadillo, Happy Buddha, ...) y
muestra el proceso **iteracion por iteracion en Polyscope**: la nube moviendose,
las correspondencias usadas, el residual por punto y la curva de convergencia.

Permite comparar, sobre exactamente el mismo problema:

| Eje de comparacion | Opciones |
|---|---|
| Busqueda de vecino mas cercano | `brute-loop` (ingenua, un bucle por consulta), `brute` (vectorizada, sigue siendo O(N·M)), `kdtree` (arbol k-d, optimizada) |
| Metrica de minimizacion | `point2point` (Besl & McKay, SVD cerrada), `point2plane` (Chen & Medioni, planos tangentes, sistema 6×6 linealizado) |

---

## Instalacion

```bash
pip install numpy scipy polyscope matplotlib
```

Los modelos 3D se descargan solos la primera vez (repositorio publico
`alecjacobson/common-3d-test-models`, que reune las mallas clasicas del Stanford
3D Scanning Repository) y quedan cacheados en `./data/`.

```bash
python run_icp.py models                 # lista los modelos
python run_icp.py models --fetch all     # los descarga todos por adelantado
```

## Uso rapido

```bash
# ICP punto a punto sobre el conejo de Stanford + visor de Polyscope
python run_icp.py demo

# variante de planos tangentes, mas puntos
python run_icp.py demo --model armadillo -n 20000 --variant point2plane

# comparacion de las 4 combinaciones (metrica x backend) + grafico
python run_icp.py compare --plot convergencia.png

# costo de la busqueda de vecinos segun el tamano de la nube
python run_icp.py bench-nn --plot escalamiento_nn.png

# solo consola, sin abrir el visor (util por ssh)
python run_icp.py demo --no-view --verbose
```

### Opciones principales

**Escena** (`demo`, `compare`)

| Opcion | Que hace |
|---|---|
| `--model` | modelo conocido o ruta a un `.obj` propio |
| `-n, --n-points` | puntos muestreados en la nube fuente |
| `--angle`, `--translation` | desalineacion inicial (ground truth conocido) |
| `--noise` | ruido gaussiano sobre la fuente |
| `--overlap` | recorta la fuente por un plano para simular solapamiento parcial |
| `--estimate-normals` | estima las normales del objetivo por PCA (como en un escaner real) en vez de tomarlas de la malla |

**Algoritmo**

| Opcion | Que hace |
|---|---|
| `--variant` | `point2point` o `point2plane` |
| `--nn` | `kdtree`, `brute` o `brute-loop` |
| `--max-iter` | tope de iteraciones |
| `--sample` | submuestreo aleatorio de la fuente en cada iteracion |
| `--reject` | umbral de rechazo = factor × mediana de distancias (0 lo desactiva) |
| `--trim` | fraccion de pares conservados por percentil (trimmed ICP) |
| `--normal-reject` | descarta pares cuyas normales difieran mas de X grados |

## El visor de Polyscope

Estructuras: **objetivo (fijo)** con sus normales, **fuente (pose inicial)**,
**fuente (ground truth)** (desactivada) y **fuente (ICP)**, coloreada por
distancia al vecino mas cercano. Las **correspondencias** se dibujan como
segmentos, en rojo/azul segun hayan sido aceptadas o rechazadas.

En el panel lateral:

- **Reproduccion**: play/pausa, paso a paso, slider de iteracion, velocidad.
- **Estado de la iteracion**: RMSE de las correspondencias, RMSE contra el
  ground truth, error de rotacion y traslacion, numero de pares aceptados,
  magnitud del paso aplicado y tiempo gastado en vecinos vs minimizacion.
- Graficos `log10 RMSE` y `log10 err. GT` por iteracion.
- **Volver a ejecutar**: cambia metrica, backend, iteraciones, submuestreo o
  umbral de rechazo y vuelve a correr ICP sin salir del visor.

## Guion sugerido para la clase

1. **El algoritmo, paso a paso.** `python run_icp.py demo -n 5000`
   Pausar en la iteracion 0 y avanzar con "paso >>": se ven las
   correspondencias cambiando y el error bajando. Es la forma mas directa de
   mostrar que ICP alterna *correspondencia* y *minimizacion*.

2. **Costo del vecino mas cercano.** `python run_icp.py bench-nn`
   La version ingenua es ~O(N²) y el kd-tree casi lineal; con N = 50.000 la
   diferencia es de ~30× en la consulta (y crece con N). El arbol se construye
   **una sola vez** porque el objetivo no se mueve: solo se consulta N veces por
   iteracion. Ese es el truco que hace practico a ICP.

3. **Punto a punto vs punto a plano.** `python run_icp.py compare --plot convergencia.png`
   Mismo problema, mismo resultado final, pero point-to-plane converge en ~10
   iteraciones donde point-to-point necesita ~50: la metrica de planos
   tangentes permite deslizar sobre la superficie sin penalizacion.

4. **Minimos locales.** `python run_icp.py compare --angle 140 --max-iter 80`
   Con una desalineacion inicial grande ICP converge a una pose incorrecta
   aunque el RMSE baje: buena excusa para hablar de inicializacion global.
   Con `--angle 100` se ve un caso intermedio donde point-to-plane acierta y
   point-to-point se queda a mitad de camino.

5. **Datos realistas.** `--noise 0.005 --overlap 0.6 --estimate-normals`
   Solapamiento parcial y normales estimadas por PCA; ahi se nota para que
   sirven `--reject` y `--trim` (sin rechazo de pares el borde del recorte
   arrastra el registro).

## Estructura del codigo

```
run_icp.py            punto de entrada
icp_demo/
  data.py             descarga de modelos, lector OBJ, muestreo por area
  geometry.py         transformaciones rigidas, Rodrigues, normales por PCA, metricas
  neighbors.py        los tres backends de vecino mas cercano
  icp.py              el algoritmo y los dos solvers (SVD y planos tangentes)
  scenario.py         construccion del problema (desalineacion, ruido, recorte)
  viz.py              visor de Polyscope con reproduccion iteracion a iteracion
  benchmark.py        comparativas y graficos
  cli.py              linea de comandos
```

Mapa rapido a las diapositivas:

| Paso del ICP | Donde esta |
|---|---|
| Busqueda de correspondencias | `neighbors.py` + `icp.correspondences` |
| Rechazo de pares | `icp.correspondences` (umbral, trimming, normales) |
| Minimizacion punto a punto | `icp.solve_point_to_point` (SVD de Arun/Horn) |
| Minimizacion punto a plano | `icp.solve_point_to_plane` (sistema normal 6×6) |
| Actualizacion y criterio de parada | bucle principal de `icp.icp` |

## Notas de implementacion

- La escala se normaliza a diagonal de bounding box = 1, asi los umbrales
  (`--noise`, `--reject`) son comparables entre modelos.
- Fuente y objetivo se muestrean **de forma independiente** sobre la malla: no
  existe la correspondencia perfecta, como en un caso real.
- El RMSE que reporta ICP se calcula con **sus propias** correspondencias, por
  eso puede ser bajo con un registro malo; por eso ademas se muestra el error
  contra el ground truth (`err_rot`, `err_trans`), que es la metrica honesta.
- Para una misma metrica, los tres backends dan resultados numericos identicos;
  lo unico que cambia es el tiempo.
- `cKDTree` consulta en paralelo (`workers=-1`); si se quiere una comparacion
  estrictamente monohilo, cambiar ese valor en `neighbors.KDTreeBackend`.
- Con `--sample` las correspondencias cambian de iteracion en iteracion, asi que
  el RMSE fluctua y el criterio de parada por tolerancia casi nunca se cumple:
  es el comportamiento esperado del ICP estocastico.
