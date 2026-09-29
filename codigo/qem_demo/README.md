# Demo: simplificación de mallas con cuádricas de error

Implementación didáctica de **Garland & Heckbert, "Surface Simplification Using
Quadric Error Metrics", SIGGRAPH 1997**, con un visor Polyscope interactivo.
Mantiene la estructura de los demos anteriores: paquete + CLI (`run_qem.py`, como
`run_icp.py` y `run_recon.py`), radio buttons en el panel, `--shot` / `--no-view`.

```
conda activate procgeom
pip install -r requirements.txt          # numpy scipy polyscope matplotlib certifi (+ open3d opcional)

python run_qem.py                        # visor con la vaca del paper, 994 caras
python run_qem.py demo --model fandisk   # pieza CAD: planos, pliegues y esquinas
python run_qem.py explain                # recorrido numérico paso a paso (consola)
python run_qem.py compare --model cow --plot curvas.png
python run_qem.py models --download      # baja cow, fandisk, bunny, armadillo
```

`cow` y `fandisk` vienen incluidos en `data/`; `bunny` y `armadillo` se bajan solos
la primera vez (mismo repositorio
`alecjacobson/common-3d-test-models` y misma cadena de descarga con certifi que
el demo de ICP; `--insecure` como último recurso). Las mallas se normalizan a
diagonal 1, así que **todos los errores están en milésimas de la diagonal (‰)**.

---

## Cómo leer el código (en este orden)

| archivo | qué tiene |
|---|---|
| `qem_demo/quadrics.py` | **Toda la matemática.** Plano p = (a,b,c,d) → K_p = p pᵀ; Q de vértice = Σ K_p; Δ(v) = v̂ᵀQv̂; v̄ = −A⁻¹b y sus alternativas; restricciones de borde; rango de A y elipsoides. El docstring de arriba es la clase en una página. |
| `qem_demo/decimate.py` | **El algoritmo.** Clase `PairCollapse` con los 5 pasos del resumen del paper (§4) con esos nombres: `step1_compute_quadrics`, `step2_select_pairs`, `step3_4_build_heap`, `simplify` (paso 5), más `is_valid_contraction` (condición de enlace), `flips_normal` y `contract`. Estructuras en listas de `set` de Python, a propósito. |
| `qem_demo/explain.py` | `run_qem.py explain`: imprime cada paso con números sobre una pirámide de 81 vértices. |
| `qem_demo/baselines.py` | Clustering de vértices (Rossignac–Borrel 1993), clustering con cuádricas (Lindstrom 2000) y Open3D. "Arista más corta" es el mismo `PairCollapse` con `cost="length"`. |
| `qem_demo/metrics.py` | Distancia exacta punto–triángulo (Ericson), Hausdorff simétrico y error medio, calidad de triángulos. |
| `qem_demo/psview.py` | Visor Polyscope. |
| `qem_demo/_compat.py` | Compatibilidad Polyscope 1.3 – 2.6 (ver abajo). |

La implementación es Python puro + numpy (sin extensiones compiladas): la vaca
completa se simplifica en ~0.7 s, el conejo en ~9 s y el armadillo (100k caras)
en ~15 s. Open3D (C++) hace lo mismo unas 30× más rápido; es un buen punto para
la clase: el algoritmo es el mismo, la diferencia es la implementación.

## El visor

Se calcula **la secuencia completa de colapsos una sola vez** y el panel la
recorre: el slider de caras, `−1 / +1`, `Play` y los botones de objetivo no
vuelven a correr el algoritmo (la malla en el paso k se reconstruye del
historial; se verificó que es idéntica a la del algoritmo incremental).

- **Modelo**: cow, fandisk, bunny, armadillo (reales) y terreno, toro, piezas (sintéticos).
- **Método**: QEM · arista más corta · clustering · clustering + cuádricas · Open3D.
  Para QEM: posición de v̄ (`optimal` del paper, `svd`, `subset` = mejor de v₁/v₂/medio,
  `midpoint`), ponderar por área, restricción de borde, preservar topología,
  rechazar inversión de normales y el umbral **t** de pares no conectados.
- **Simplificación**: slider de caras, ±1 colapso, Play, objetivos 50/20/5/1 % y,
  para la vaca, **994 / 532 / 248 / 64** (las caras de la figura 5 del paper).
  Gráfico del costo de cada colapso (log10) y cuántos v̄ salieron de A⁻¹b,
  del segmento o del "mejor de tres".
- **Próximo colapso**: la arista que saldrá del heap (v₁ azul sobrevive, v₂ rojo
  desaparece, v̄ amarillo), su costo, cómo se eligió v̄, los autovalores de A del par
  y el elipsoide de Q₁+Q₂. "Acercar cámara" va a la arista.
- **Visualización**: tipo de vértice por rango de A (plano / pliegue / esquina) como
  esferitas, error de la cuádrica √(Δ/w) (distancia RMS a los planos que
  representa el vértice), distancia a la original, calidad de triángulos, y los
  **elipsoides de error** de todos los vértices (disco = plano, cigarro =
  pliegue o borde, esfera = esquina).
- **Resultados**: χ, componentes, bordes, género, Hausdorff / medio / RMS (‰),
  calidad; "comparar todos los métodos" llena una tabla con el número de caras
  actual; "guardar OBJ" escribe en `salidas/`.

Sin pantalla (`--shot PREFIJO`) guarda un PNG por estado (rango, elipsoides,
distancia, próximo colapso, arista corta, clustering).

---

## Guion sugerido para la clase

Los números son de esta implementación (Linux, 2 núcleos); se reproducen con los
comandos indicados.

### 1. La idea en una malla de juguete — `python run_qem.py explain`

1. **Plano → cuádrica.** Muestra p, K_p = p pᵀ y comprueba que x̂ᵀK_p x̂ = (n·x + d)².
2. **Rango de A.** En la pirámide: piso → autovalores (0, 0, λ) = *plano*; base de la
   pirámide → (0, λ₂, λ₃) = *pliegue*; cúspide y esquinas → tres autovalores = *esquina*.
   Un vértice del borde pasa de *plano* a *pliegue* al agregar el plano perpendicular
   al borde: ya solo puede deslizarse **a lo largo** del borde.
3. **Costo de un par.** Piso–piso: cuesta 0 en cualquier punto. Piso→pliegue: el óptimo
   (en el segmento) cuesta 0 y el punto medio 1.0e-3. Cúspide→ladera: A invertible,
   v̄ = −A⁻¹b cuesta 1.6e-3, el mejor extremo 2.0e-3 y el punto medio 3.7e-3.
4. **El heap.** 176 de 208 pares cuestan 0.
5. **El momento clave:** los primeros **64 colapsos cuestan 0** y dejan 17 vértices
   y 28 caras que describen **exactamente** la misma forma; el colapso 65 ya cuesta
   7.8e-4. Las cuádricas "saben" qué vértices sobran.

### 2. La vaca del paper — `python run_qem.py`

Botones 994 / 532 / 248 / 64. Mostrar "próximo colapso" con ±1 y el gráfico de
costos (crece varios órdenes de magnitud). "Comparar todos los métodos":

| 994 caras | Hausdorff ‰ | medio ‰ |
|---|---|---|
| QEM, v̄ óptimo | 15.4 | **0.76** |
| QEM, mejor de v₁/v₂/medio | 19.2 | 1.05 |
| QEM, punto medio | 22.5 | 1.50 |
| arista más corta | 34.5 | 2.54 |
| clustering (promedio), 1041 caras | 30.6 | 2.47 |
| clustering + cuádricas, 1041 caras | 23.0 | 1.56 |
| Open3D | 15.4 | 0.76 |

A 64 caras: QEM 9.3, arista más corta 25.1, clustering 25.1 (medio ‰). Lectura:
la **métrica** decide el orden (QEM-punto-medio ya le gana a arista más corta)
y la **posición** v̄ óptima baja el error otro ~2×. El clustering deja la malla
no-variedad (χ = 9, género indefinido). `compare --plot` dibuja la curva
error vs caras de todos los métodos.

### 3. Qué "ve" la cuádrica — `python run_qem.py demo --model fandisk`

Color "tipo de vértice" + "elipsoides de error": discos en las caras planas,
cigarros a lo largo de las aristas vivas, esferas en las esquinas (la figura de
elipsoides del paper). Con Play: primero desaparecen los vértices de las caras
planas — **4963 colapsos de costo 0** llevan de 12946 a 3020 caras sin error.

| fandisk | 1000 caras (medio ‰) | 300 caras | 100 caras |
|---|---|---|---|
| QEM | **0.03** | **0.19** | **1.8** |
| arista más corta | 3.1 | 8.8 | 19.0 |
| clustering + cuádricas | 0.40 | 1.7 | 8.4 |

El precio: triángulos largos (calidad media 0.51 vs 0.90 de arista más corta).
QEM optimiza error geométrico, no forma de triángulos.

### 4. Bordes — `demo --model terreno` (y `--model bunny`)

Desmarcar "restricción de borde":

| terreno | con borde | sin borde |
|---|---|---|
| 1000 caras, Hausdorff ‰ | 1.9 | **186** |
| 300 caras, Hausdorff ‰ | 7.3 | **186** |

Sin los planos perpendiculares el borde no cuesta nada y se come hacia adentro.
En el conejo (5 hoyos): a 2000 caras, Hausdorff 11.6 → 34.4 y los hoyos se
cierran de 143 a 39 aristas de borde.

### 5. Topología — `demo --model toro` y la vaca

Con "preservar topología" el toro llega a 16 caras con género 1. Sin ella, a 18
caras aparece una arista no-variedad y el género deja de estar definido. En la
vaca sin la condición de enlace: a 994 caras ya hay 2 aristas no-variedad, a 248
caras la vaca se parte en 2 componentes (patas/cuernos finos se "pellizcan").

### 6. Pares no conectados (umbral t) — `demo --model piezas`

16 bloques separados por una ranura. A 192 caras todos los métodos son exactos
(cada bloque = 12 triángulos). Sin pares virtuales el mínimo es 64 caras (16
tetraedros). Con **t = 0.015 y sin preservar topología** los bloques se funden en
una sola componente y el error medio baja: 150 caras 4.9 → 3.0 ‰, 100 caras
14.7 → 9.0 ‰, ~64 caras 24.9 → 18.7 ‰.

### 7. Detalle numérico que vale la pena contar

¿Cuándo es "A invertible"? Aquí: λ_min(A) > 10⁻⁵ · λ_max(A), que no depende de la
escala. Open3D usa un umbral **absoluto** sobre det(A): con la vaca normalizada a
diagonal 1, det(A) nunca lo supera y Open3D cae siempre en "mejor de v₁/v₂/medio"
(Hausdorff 19.2 ‰, igual que nuestro `subset`); escalando la malla ×100 da 15.4 ‰,
igual que el óptimo. Por eso el demo llama a Open3D con la malla escalada ×1000.

---

## Decisiones de implementación

- **Heap con borrado perezoso.** Al contraer, los pares viejos no se sacan del heap:
  cada vértice tiene un número de versión y las entradas desactualizadas se
  descartan al salir (contador "entradas viejas del heap").
- **Colapso rechazado = par descartado** hasta que uno de sus vértices cambie
  (QSlim penaliza en vez de descartar).
- **Condición de enlace** (Dey et al. 1999) para preservar topología, con el caso de
  borde (dos vértices de borde unidos por una arista interior) y el triángulo suelto.
- **Sin preservar topología**, dos caras que quedan con los mismos 3 vértices se
  borran ambas (son paredes internas al fundir piezas).
- **Rango de A y elipsoides** se miden contra el peso de las caras del vértice
  (su área), no contra λ_max: con restricción de borde λ_max es enorme y todo lo demás
  parecería cero.
- **Vértices pellizcados** se separan al cargar: la vaca del repositorio tiene uno
  (sin separarlo χ = 1 y el género sale 0.5).
- **Distancias exactas**: muestras que cubren cada triángulo + distancia exacta a los
  triángulos candidatos; se verificó contra fuerza bruta (error 0).

## Compatibilidad

Probado con Polyscope **1.3.4, 2.0.0, 2.1.0, 2.3.0 y 2.6.1**:

- `SeparatorText` y `BeginTable` tienen reemplazo (texto / tabla en texto plano).
- Polyscope < 2 no tiene `frame_tick`: se usa `show(forFrames=n)`.
- **La fuente de Polyscope ≤ 2.3 solo trae Latin-1**: χ, λ, v̄, ‰ salían como "?".
  Todo el texto del panel pasa por `_compat.T()`, que los escribe como chi, lambda,
  vbar, o/oo. Las tildes y la ñ se ven bien en todas las versiones.
- La consola de Windows con página de códigos antigua no rompe los `print` con
  símbolos (se reemplazan).
