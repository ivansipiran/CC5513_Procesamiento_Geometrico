# Demo: reconstruccion de superficies (SGP / CC5513, clase 7)

Demo interactivo que acompana al tutorial de la Clase 7 (normales y reconstruccion).
Mismos metodos, misma convencion de signo (**F < 0 dentro, F > 0 fuera**) y la misma
superficie implicita de genero 1, pero en 3D y en vivo con **Polyscope**:

1. **Nube de entrada** -- blob de genero 1 (F exacta) o un modelo de Stanford, con ruido,
   outliers, un hueco o baja densidad.
2. **Normales** -- PCA local + orientacion (MST de Hoppe, centroide o ninguna) y filtro
   estadistico de outliers.
3. **Campo implicito** -- Hoppe 1992, RBF de Carr 2001 (phi(r) = r) y Poisson por FFT;
   opcionalmente Poisson screened de Open3D como referencia de produccion.
4. **Extraccion** -- marching tetrahedra propio (el del tutorial) o marching cubes de
   scikit-image.
5. **Metricas** -- error contra la superficie real, completitud (detecta huecos),
   chi = V - E + F y genero.

## Instalacion

```bash
conda activate procgeom
pip install -r requirements.txt      # numpy scipy polyscope certifi (+ opcionales)
```

`scikit-image` (marching cubes) y `open3d` (Poisson screened) son opcionales: si no
estan, esas opciones simplemente no aparecen.

Los modelos reales se descargan solos la primera vez desde
`alecjacobson/common-3d-test-models` a la carpeta `data/`. Si la red de la sala falla:

```bash
python run_recon.py models --fetch all            # descarga todo por adelantado
python run_recon.py models --fetch all --insecure # si el certificado TLS da problemas
```

## Uso rapido

```bash
python run_recon.py                                # = demo: blob limpio + visor
python run_recon.py demo --preset hueco            # regimenes: limpio ruido hueco
                                                   #   "densidad baja" outliers
python run_recon.py demo --model bunny -n 15000
python run_recon.py demo --normals none            # normales sin orientar
python run_recon.py compare                        # tabla metodos x regimenes
python run_recon.py compare --model bunny --csv bunny.csv
python run_recon.py normals                        # error de normales vs k y ruido
python run_recon.py demo --no-view                 # solo consola
python run_recon.py demo --shot capturas/blob      # sin pantalla: PNGs y termina
```

Opciones principales de `demo` / `compare`:

| Opcion | Que hace |
|---|---|
| `--model` | `blob`, `bunny`, `armadillo`, `planck`, `fandisk` |
| `--preset` | regimen predefinido (sobrescribe las 4 opciones siguientes) |
| `-n`, `--noise`, `--outliers`, `--hole` | puntos, sigma del ruido, fraccion de outliers, radio del hueco. **Ruido y hueco en fraccion de la diagonal**, asi los numeros sirven para cualquier modelo |
| `--normals` | `mst` (defecto), `centroid`, `none`, `true` (verdaderas) |
| `-k`, `--filter`, `--alpha` | vecinos de PCA; filtro de outliers (media + alpha * desv) |
| `--methods` | subconjunto de `Hoppe RBF Poisson Open3D` |
| `--res` | celdas de la grilla en el eje mas largo (defecto 64) |
| `--extractor` | `mt` (marching tetrahedra propio) o `mc` (scikit-image) |
| `--hoppe-support` | rho + delta de Hoppe, en multiplos del espaciado medio |
| `--rbf-centers`, `--rbf-eps` | centros de la RBF y desplazamiento de las restricciones |
| `--poisson-sigma` | suavizado gaussiano del campo V (en celdas) |

## El visor

Estructuras: **nube de entrada** (coloreable por error angular, orientacion -- verde bien,
rojo invertida --, variacion de superficie u outliers reales; con normales estimadas y
verdaderas), **descartados por el filtro** (verde = outlier real, naranja = inlier
perdido), **superficie real** (desactivada), una malla por metodo coloreada por error, y
el **corte del campo** (azul dentro, rojo fuera, gris = no definido, con curvas de nivel)
con su **iso-linea F = 0** calculada por marching squares.

Panel lateral:

1. **Nube de entrada** -- modelo, botones de regimen, sliders de ruido / outliers / hueco.
2. **Normales** -- metodo de orientacion, k, filtro; muestra error angular mediano,
   % bien orientadas, precision / recall del filtro.
3. **Reconstruccion** -- metodos activos, resolucion, extractor y los parametros de cada
   metodo. Solo se recalcula lo que cambio (cambiar el eps de la RBF no rehace Poisson).
4. **Resultados** -- que malla ver (o **lado a lado**), tabla de errores, V/E/F y chi del
   metodo activo, y datos internos: % de grilla sin definir en Hoppe, tamano del sistema
   de la RBF, iso-valor de Poisson.
5. **Corte del campo** -- eje y posicion; superpone el **campo V** de Poisson y las
   **restricciones de la RBF** (azul -eps, blanco 0, rojo +eps) o los centros de los
   planos de Hoppe. Con el corte activo, la malla se vuelve semitransparente.

Las columnas de la tabla, en milesimas de la diagonal:
**err** = distancia de los vertices a la superficie real (mediana y p95);
**compl p95** = distancia de la superficie real a la malla (la precision no penaliza que
falte superficie; esta si); **cobert.** = fraccion de la superficie real a < 2 % de la
diagonal de la malla.

## Guion sugerido para la clase

Numeros del blob, 8000 puntos, grilla 64, semilla 0 (`python run_recon.py compare`).

1. **Normales y orientacion.** `python run_recon.py demo`, seccion 2, colorear por
   *orientacion*. Con MST: 100 % bien orientadas. Con *hacia afuera del centroide*: 84 %
   -- falla justo en la cara interior del anillo, que es la razon de ser del MST en una
   superficie de genero 1. *Sin orientar*: 50 %, y las tres reconstrucciones se destruyen
   (el error mediano de Poisson pasa de 0.5 a 48 milesimas y la malla queda abierta). La orientacion no es un detalle.

2. **El compromiso de k.** `python run_recon.py normals`: con ruido, k pequeno amplifica
   el ruido y k grande promedia a traves de la curvatura. k* = 20, 30, 60, 90, 130 para
   sigma/diag = 0, 0.001, 0.002, 0.004, 0.008 -- el mismo patron del bloque 1.3.

3. **Marching tetrahedra y topologia.** Seccion 4: chi = V - E + F = 0 para las tres
   mallas -> genero 1, igual que la superficie real. Marcar *aristas* y alternar el
   extractor MT / MC para comparar teselados: con Poisson, MT da 58k triangulos y MC
   19k (3x mas y de peor forma), a cambio de 16 casos sin tabla de 256 ni ambiguedad.

4. **Que funcion construye cada metodo.** Seccion 5, corte en z. Hoppe: una distancia
   con signo (curvas de nivel equiespaciadas), definida solo cerca de los datos. RBF:
   activar *restricciones* para ver los pares +-eps. Poisson: activar *campo V* -- el
   campo es cero lejos de la superficie y la funcion es casi constante dentro y fuera:
   una funcion indicadora suavizada, no una distancia.

5. **Regimenes** (botones de la seccion 1, o `compare`):

   | regimen | Hoppe | RBF | Poisson | lectura |
   |---|---|---|---|---|
   | limpio | 0.83 | 0.66 | 0.53 | empate: los tres genero 1 |
   | ruido (sigma = 0.53 % diag) | 1.17 | **3.17** | **0.86** | la RBF interpola el ruido; Poisson lo integra |
   | hueco (tapa de radio 24 % diag, 16 % de los puntos) | abierta, compl 62 | **compl 15** | compl 23 | la RBF rellena mejor que Poisson |
   | densidad baja (1200 pts) | **4.96** | **0.64** | 0.97 | Hoppe es lineal a trozos |
   | outliers 6 % + filtro | 0.83 | p95 20, **14 comp.** | 0.60 | la RBF interpola lo que el filtro dejo |

   (error mediano en milesimas de la diagonal, salvo donde se indica.)
   El **hueco** es una tapa que el escaner no vio, sobre el tubo del anillo (solo se
   quitan los puntos que miran hacia arriba: la cara de abajo sigue ahi). Con el corte en
   x a la altura del hueco (posicion ~0.2) y Hoppe activo, la zona gris es donde la
   funcion no esta definida: por eso la malla queda abierta. Cambiando a RBF y Poisson se
   ve la diferencia de forma: la RBF tiende un puente curvo (el spline biarmonico
   extrapola curvatura) y Poisson uno aplanado (donde V = 0 la solucion es armonica, lo
   mas plana posible). La intuicion "Poisson siempre gana" solo vale con ruido.
   Ojo: en el conejo (`--model bunny --preset hueco`) el orden se invierte (Poisson
   compl 22, RBF 30) -- depende de la geometria que falta, buen tema de discusion.

6. **Hoppe y el ruido.** Regimen *ruido* y bajar `k (centroide del plano)` de 20 a 3:
   aparecen tuneles espurios -- genero 16 en vez de 1 -- porque el plano de un solo vecino
   sigue al ruido. El centroide de k vecinos es un filtro.

7. **Datos reales.** `--model bunny`: el conejo tiene agujeros en la base. Hoppe los
   respeta (malla abierta); Poisson y RBF los cierran (genero 0). En las orejas, que son
   delgadas, Hoppe elige el plano de la cara opuesta y aparecen puntas: el problema de
   las estructuras mas delgadas que rho + delta. `--model fandisk` para aristas vivas.
   Open3D (octree, B-splines y termino de screening) queda como referencia de produccion.

## Notas de implementacion

- **Poisson por FFT** sobre la grilla regular con un margen del 10 %: la frontera es
  periodica, asi que la funcion necesita "aire" alrededor. Las normales se pesan por el
  area local que representa cada muestra (r_k^2) para no sesgar el campo en zonas densas.
- **RBF**: centros por muestreo del punto mas lejano, sistema denso (3m + 4) resuelto
  directo. El chequeo de Carr para eps se hace como "p + eps n no debe quedar a menos de
  eps/2 de otra muestra": la regla literal ("el mas cercano debe ser p") encoge eps hasta
  el nivel del ruido y deja las restricciones sin efecto.
- **Hoppe**: planos por el centroide de k vecinos; indefinido si la proyeccion al plano
  queda a mas de rho + delta de la nube (delta = 2 sigma del ruido conocido).
- **Marching tetrahedra** vectorizado: solo se procesan las celdas con cambio de signo;
  los vertices se identifican por arista de la grilla, asi la malla sale indexada y chi
  es exacto.
- La **distancia a la superficie real** es |F| / |grad F| para el blob y, para los
  modelos, la distancia al plano tangente de la muestra mas cercana de una nube densa de
  400k puntos.

## Compatibilidad

Probado con Polyscope 1.3.4, 2.0.0, 2.1.0 y 2.6.1 (numpy 1.26 y 2.x). Las funciones de
imgui que no existen en versiones antiguas (`SeparatorText`, tablas) tienen reemplazo.
Si al abrir el visor aparece un error de imgui, basta con copiar el mensaje completo.

## Estructura

```
run_recon.py          punto de entrada (agrega su carpeta a sys.path)
sr_demo/data.py       blob de genero 1, modelos, escenario, regimenes
sr_demo/normals.py    PCA, MST, centroide, filtro de outliers
sr_demo/grid.py       grilla regular
sr_demo/methods.py    Hoppe, RBF, Poisson (FFT), Open3D
sr_demo/extract.py    marching tetrahedra, marching cubes, marching squares
sr_demo/metrics.py    chi, genero, bordes, precision y completitud
sr_demo/pipeline.py   orquestacion y formato de la tabla
sr_demo/viz.py        visor de Polyscope
sr_demo/cli.py        demo / compare / normals / models
```
