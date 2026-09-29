# Global Registration — implementaciones para la Clase 6 (SGP)

Dos caminos completos para global registration, implementados desde cero sobre
numpy/scipy (Open3D solo se usa para I/O, kd-trees, ICP y como baseline):

1. **Con features — la receta de las slides**
   Harris 3D (Sipiran & Bustos 2011) → spin images (Johnson & Hebert 1999) →
   matching por correlación → agrupamiento geométrico → ICP.
2. **Sin features** — 4PCS (Aiger et al. 2008) y Super4PCS (Mellado et al. 2014).

## Instalación

```bash
pip install -r requirements.txt
```

## Uso rápido

```bash
python3 demo.py --mesh bunny --dphi 90                    # los cuatro métodos
python3 demo.py --mesh armadillo --dphi 120               # aquí falla la receta
python3 demo.py --mesh armadillo --dphi 135 --method super4pcs
python3 demo.py --dphi 90 --no-view                       # solo los números
```

`--dphi` es el ángulo entre las dos vistas simuladas y controla el solapamiento
(45° ≈ 0.87, 90° ≈ 0.65, 120° ≈ 0.37, 135° ≈ 0.30).

El demo corre los métodos y después abre **Polyscope** con una sola escena: la
nube destino en azul, la nube origen en naranja, y las correspondencias que
produjeron Harris 3D + spin images en verde (correctas) y rojo (incorrectas).
Los radio buttons del panel cambian la pose del origen entre la configuración
inicial, el resultado de cada método y el ground truth; el panel muestra el
error de cada uno. Un checkbox muestra los keypoints.

En una máquina sin pantalla (contenedor, servidor por SSH) Polyscope arranca con
el backend EGL headless y guarda un PNG por estado en vez de abrir la ventana;
`--shot PREFIJO` fuerza ese modo. `--no-view` salta la visualización, y `--fig
out.png` guarda además una figura estática de matplotlib.

## Estructura

| Archivo | Contenido |
|---|---|
| `gr/data.py` | Banco de pruebas: range scans simulados por raycasting desde dos vistas, transformación rígida aleatoria como ground truth exacto, ruido y outliers. |
| `gr/harris3d.py` | Harris 3D completo: vecindades adaptativas por anillos (mallas) y kNN (nubes), traslación al origen, PCA, ajuste cuadrático, matriz de auto-correlación, respuesta y selección con supresión de no máximos. |
| `gr/spin_images.py` | Spin images con acumulación bilineal y ángulo de soporte; similitud `C(P,Q) = atanh(R)² − λ/(N−3)` sobre los bins comunes; matching mutuo / top-k. |
| `gr/grouping.py` | Umeyama/Horn, criterio `W_gc` de Johnson & Hebert (el de la slide 26), clique por consistencia de distancias, RANSAC y verificación LCP. |
| `gr/pipeline_feature.py` | La receta completa, con detector / descriptor / solver intercambiables. |
| `gr/fourpcs.py` | 4PCS y Super4PCS: `SelectCoplanarBase`, `FindCongruent`, extracción de pares por cáscara rasterizada, join por grilla y filtro por normales. |
| `gr/baselines.py` | FPFH+RANSAC y FGR de Open3D. |
| `gr/metrics.py` | Error de rotación/traslación, RMSE contra ground truth, repetibilidad de keypoints, razón de inliers. |
| `gr/viz.py` | Figuras estáticas (matplotlib). |
| `gr/psview.py` | Visor interactivo con Polyscope: estados intercambiables, correspondencias coloreadas por inlier/outlier, keypoints, y captura headless por EGL. |

## Experimentos

```bash
python3 experiments/exp1_repeatability.py   # ¿los keypoints son repetibles?
python3 experiments/exp2_descriptors.py     # ¿qué tan buenas son las correspondencias?
python3 experiments/exp3_full.py            # comparación completa (8 métodos)
python3 experiments/exp4_scaling.py         # 4PCS vs Super4PCS: escalamiento
python3 experiments/exp5_solver.py          # dónde se rompe la receta
python3 experiments/exp6_real.py            # fragmentos RGB-D reales
python3 experiments/make_figures.py         # figuras del pipeline
```

Los resultados quedan en `experiments/results_*.json` y las figuras en `figs/`.

## Resultado corto

La receta **sí** se puede implementar y **sí** da resultados. Con 30 pares con
ground truth exacto (bunny y armadillo, solapamiento 0.28–0.87, con y sin ruido):

| Método | recall global | ov > 0.8 | ov ≈ 0.6 | ov ≈ 0.3 | ruido | tiempo |
|---|---|---|---|---|---|---|
| Harris3D + spin + **agrupamiento (slides)** | 0.50 | 1.00 | 1.00 | 0.00 | 0.25 | 0.8 s |
| Harris3D + spin + **RANSAC** | **0.93** | 1.00 | 1.00 | **0.83** | 0.92 | 0.7 s |
| Harris3D + FPFH + RANSAC | 0.87 | 1.00 | 1.00 | 0.50 | 0.92 | 0.7 s |
| Muestreo uniforme + spin + RANSAC | 0.83 | 1.00 | 1.00 | 0.17 | 1.00 | 0.5 s |
| FPFH + RANSAC (Open3D) | 0.90 | 1.00 | 1.00 | 0.50 | 1.00 | 0.8 s |
| FGR (Open3D) | 0.87 | 1.00 | 1.00 | 0.33 | 1.00 | 0.4 s |
| 4PCS | 0.87 | 1.00 | 1.00 | 0.33 | 1.00 | 8.0 s |
| Super4PCS | 0.87 | 1.00 | 1.00 | 0.33 | 1.00 | 9.9 s |

El detector y el descriptor de las slides están bien; el eslabón débil es el
**paso final de agrupamiento**. Cambiándolo por RANSAC sobre las mismas
correspondencias, la receta pasa de 0.50 a 0.93 de recall. Con solapamiento alto
todos los métodos empatan y las diferencias de ±0.05 en el recall global son una
o dos corridas de 30 — ruido. La separación real está en solapamiento bajo:
0.83 contra 0.33–0.50 del resto. El detalle está en el reporte.

## Referencias

- I. Sipiran, B. Bustos. *Harris 3D: a robust extension of the Harris operator
  for interest point detection on 3D meshes.* The Visual Computer 27(11), 2011.
- A. Johnson, M. Hebert. *Using spin images for efficient object recognition in
  cluttered 3D scenes.* IEEE TPAMI 21(5), 1999.
- D. Aiger, N. Mitra, D. Cohen-Or. *4-points congruent sets for robust pairwise
  surface registration.* ACM SIGGRAPH 2008.
- N. Mellado, D. Aiger, N. Mitra. *Super 4PCS: fast global pointcloud
  registration via smart indexing.* Computer Graphics Forum (SGP) 33(5), 2014.
