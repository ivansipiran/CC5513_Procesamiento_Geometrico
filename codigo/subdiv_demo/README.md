# Demo: subdivisión de mallas (Catmull-Clark, Loop, Doo-Sabin)

Misma mecánica que los demos anteriores (ICP, reconstrucción, QEM): paquete Python + CLI,
visor Polyscope con radio buttons, `--shot` / `--no-view`, y un `explain` que muestra todo
con números en la consola.

```
pip install -r requirements.txt          # numpy, scipy, polyscope, certifi
python run_subdiv.py                     # visor: cubo, Catmull-Clark, nivel 2
python run_subdiv.py explain             # los tres esquemas con números
python run_subdiv.py compare --model cube --levels 4
python run_subdiv.py models --download   # baja suzanne, fandisk, cow, spot
```

## La idea que organiza el código

Cada esquema está escrito en dos partes, con los mismos nombres en los tres:

1. **Topología**: qué vértices nuevos hay y cómo se conectan.
2. **Geometría**: una matriz dispersa `S` tal que `P_nuevo = S @ P_viejo`.

La matriz *es* el esquema. Una **fila** de `S` es la máscara (el estencil) de un vértice
nuevo. Una **columna** dice a qué vértices nuevos afecta un vértice viejo (el control
local). El producto `S_k ⋯ S_1` lleva la malla de control al nivel k. Además cada paso
guarda `S_lin` (dividir sin promediar), así que el visor puede mostrar la separación
*split → average*.

| | Catmull-Clark | Loop | Doo-Sabin |
|---|---|---|---|
| tipo | primal, quads | primal, triángulos | dual, corte de esquinas |
| vértices nuevos | vértice + arista + cara | vértice + arista | uno por esquina |
| caras nuevas | un quad por esquina | 4 por triángulo | caras F + E + V |
| regular | B-spline bicúbica (C2) | box-spline cuártica (C2) | B-spline bicuadrática (C1) |
| extraordinario | vértices de valencia ≠ 4 | vértices de valencia ≠ 6 | **caras** de grado ≠ 4 |

## Archivos

```
run_subdiv.py            CLI: demo | compare | explain | models
subdiv_demo/
  mesh.py                malla poligonal CSR + half-edges vectorizados (nxt, prv, twin,
                         edge_of), abanicos de vértice, pliegues, métricas geométricas
  schemes.py             catmull_clark(), loop(), doo_sabin(): topología + S + S_lin
  limit.py               máscaras de límite (Loop, CC), matriz local 1-anillo, autovalores
  metrics.py             V/E/F, chi, irregulares, diedro, volumen, distancia al límite
  explain.py             salida didáctica en consola
  viewer.py              visor Polyscope
  models.py              mallas de control sintéticas + descarga de modelos
  download.py            descarga robusta (certifi → sistema → requests → --insecure)
  _compat.py             shims para Polyscope 1.x / 2.0–2.3 (SeparatorText, BeginTable,
                         frame_tick, fuente Latin-1 → T())
```

## Reglas implementadas

**Catmull-Clark**: punto de cara `F` = promedio de la cara; punto de arista
`(a + b + F1 + F2)/4`; punto de vértice `Q/n + 2R/n + (n−3)P/n`.
**Loop**: impar `3/8 (a+b) + 1/8 (c+d)`; par `(1 − nβ)P + β Σ vecinos`, con el β de
Loop o el de Warren (`3/8n`). Si la malla tiene polígonos se triangula antes (eligiendo
diagonales que no dupliquen aristas).
**Doo-Sabin**: esquina i de una cara de n lados `Σ α_|i−j| P_j`, con
`α_0 = (n+5)/4n`, `α_k = (3 + 2cos(2πk/n))/4n`, o la variante simple
`(P + F + M_ant + M_sig)/4`. En quads las dos dan 9/16, 3/16, 1/16, 3/16.

**Bordes y pliegues** (CC y Loop, Hoppe et al. 1994 simplificado): en aristas de borde o
marcadas como pliegue se usan reglas de curva: punto de arista = punto medio, vértice
`3/4 P + 1/8 (a + b)`, esquinas fijas. `--crease ANG` marca las aristas con diedro > ANG.
**Doo-Sabin en el borde**: `--ds-boundary chaikin` (por defecto) pone las esquinas vecinas
a una arista de borde en `3/4 a + 1/4 b`, así el borde es un corte de Chaikin;
`free` usa la regla interior y el borde se encoge. Doo-Sabin no usa pliegues en este demo.

**Límite**: Loop `(1 − nχ)P + χ Σ vecinos`, `χ = 1/(3/(8β) + n)`; CC
`(n²P + 4 Σ vecinos + Σ diagonales)/(n(n+5))`; curva `2/3 P + 1/6 (a + b)`. Verificado
contra 6 niveles de subdivisión (diferencias de 10⁻⁵).

## El visor

- **Malla de control**: 15 modelos, pliegues por ángulo, jaula (nivel 0 o nivel anterior).
- **Esquema**: CC / Loop / Doo-Sabin, variante, nivel (0–6, tope 700 k caras),
  deslizador *dividir → promediar* con **Play**, proyectar al límite, *los tres lado a lado*.
- **Colores**: tipo de vértice (esferas), tipo de cara, valencia, defecto angular.
- **Estencil (fila de S)**: elija un vértice nuevo i (o *sig. punto de cara / arista /
  vértice / extraordinario*): se dibujan los vértices del nivel anterior de los que sale,
  con sus pesos como fracciones.
- **Control local (columna de S)**: la función base de un vértice de control j sobre el
  nivel actual y un deslizador que lo mueve en su normal.
- **Resultados**: V, F, chi, irregulares, diedro máximo, volumen por nivel.

Opciones de `demo`: `--model --scheme --variant --level --crease --ds-boundary --color
--morph --limit --stencil i --basis j --delta d --side-by-side --no-cage --export x.obj
--shot x.png --ui --no-view`.

## Guion de clase sugerido (≈ 50 min, con los números que salen)

1. **`explain`** (5 min). Las tres reglas con números en el cubo y el octaedro; la tabla
   de β y autovalores de Loop; conteos de V, E, F.

2. **Topología primero, geometría después.** `python run_subdiv.py --model cube --level 1 --morph 0`
   y luego **Play**. Con el deslizador en 0 se ve solo la división (puntos medios y
   centroides); al avanzar, cada vértice se mueve a su promedio. Colores *tipo de vértice*:
   8 de vértice + 12 de arista + 6 de cara = 26.

3. **Estencil.** `--model cube --level 1 --stencil 8`. Punto de arista del cubo:
   3/8, 3/8, 1/16 × 4. En el toro (malla regular) el punto de vértice es
   9/16, 3/32 × 4, 1/64 × 4: la máscara de la B-spline bicúbica. Todos los pesos son ≥ 0 y
   suman 1, así que la superficie queda dentro de la envolvente convexa de la jaula.

4. **Los tres lado a lado.** `--model L --side-by-side --level 2 --color valencia`.
   Cubo, nivel 4 (`compare --model cube`):

   | | CC | Loop | Doo-Sabin |
   |---|---|---|---|
   | caras | 1536 | 3072 | 1538 |
   | volumen / control | 32.9 % | 37.5 % | **63.0 %** |
   | diedro máx. | 7.1° | 6.5° | 10.6° |
   | vértices / caras irregulares | 8 / 0 | 6 / – | **0 / 8** |

   Doo-Sabin (cuadrático) encoge menos y es menos suave; en Doo-Sabin lo extraordinario
   son las caras (los 8 triángulos de las esquinas del cubo), no los vértices.

5. **Convergencia.** `compare --model prism5 --levels 4`: distancia al límite, en ‰ de la
   diagonal: 165 → 40 → 10 → 2.6 (CC). Cae ~4 veces por nivel porque el largo de arista se
   divide por 2 y el error es O(h²). Active *proyectar al límite*: los vértices saltan a su
   posición final con una sola máscara (el vector propio izquierdo de λ = 1).

6. **Control local.** `--model torus --level 3 --basis 0 --delta 0.15`. La función base de un
   vértice regular cubre 4 × 4 caras de la jaula (la de la B-spline bicúbica); su máximo es
   < 1: el vértice de control no se interpola. *sig. extraordinario* en `L` o `frame` muestra
   una función base de valencia 3 o 5.

7. **El β de Loop.** `compare --model icosa --schemes loop --variants loop warren linear`
   (todo valencia 5), nivel 4: diedro máximo 3.3° con β de Loop, 4.8° con Warren. En
   `explain`, con β de Loop el autovalor subdominante es doble, λ₁ = λ₂ = 3/8 + ¼cos(2π/n),
   y λ₁ > |λ₃|: esa es la condición de C1. La variante `linear` deja el volumen en 100 % y el
   diedro en 41.8°: sin promediar, subdividir no suaviza nada.

8. **Pliegues.** `--model can --crease 60 --level 3`: los bordes de las tapas quedan vivos
   (curvas B-spline cúbicas). Volumen 90.5 % con pliegues contra 49.6 % sin ellos, y el diedro
   máximo se queda en 90°. En `fandisk --crease 40` se ve lo mismo en una pieza real.

9. **Bordes en Doo-Sabin.** `--model patch --scheme doo-sabin --level 3` y cambie *borde*.
   Distancia al límite en el nivel 3: 0.46 ‰ con Chaikin contra 8.75 ‰ con la regla
   interior, que además converge 2× por nivel en vez de 4×: el borde se va encogiendo.

10. **Suzanne** (468 quads + 32 triángulos, con bordes). Con CC el nivel 1 ya es todo quads.
    Cerca de la nariz hay un vértice interior de **valencia 2**: la regla de CC con n = 2
    tiene un peso negativo y el diedro máximo no baja (86° → 52° → 70° entre los niveles 2 y 4).
    Es un ejemplo real de por qué los modeladores evitan esas configuraciones.

## Compatibilidad

Probado con Polyscope 2.6.1 (numpy 2) y 1.3.4, 2.0.0, 2.3.0 (numpy 1.x), en modo headless.
Todo el texto del panel pasa por `_compat.T()` (la fuente de Polyscope ≤ 2.3 es Latin-1).
