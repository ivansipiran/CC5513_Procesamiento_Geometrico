"""Demo docente del algoritmo ICP (Iterative Closest Point).

Modulos:
    data       -- descarga / carga de modelos 3D reales y muestreo de nubes
    geometry   -- utilidades de rotaciones, normales y metricas de error
    neighbors  -- backends de busqueda de vecino mas cercano (naive vs kd-tree)
    icp        -- nucleo del algoritmo (point-to-point y point-to-plane)
    scenario   -- construccion del problema de registro (desalineacion, ruido, recorte)
    viz        -- visualizacion interactiva en Polyscope
    benchmark  -- comparativas de rendimiento y convergencia
"""

__version__ = "1.0"
