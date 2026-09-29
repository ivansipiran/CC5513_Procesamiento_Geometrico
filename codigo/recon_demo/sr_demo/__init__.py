"""Demo docente de reconstruccion de superficies a partir de nubes de puntos.

Sigue el tutorial de la Clase 7 (normales y reconstruccion):

    data      -- superficie implicita de referencia (blob de genero 1), modelos
                 reales descargados y construccion de la nube (ruido, outliers,
                 hueco, densidad)
    normals   -- normales por PCA local, orientacion (MST / centroide / ninguna),
                 filtro estadistico de outliers
    grid      -- grilla regular donde se evaluan las funciones implicitas
    methods   -- Hoppe 1992, RBF de Carr 2001, Poisson (FFT) y, opcional,
                 Poisson screened de Open3D
    extract   -- marching tetrahedra (propio), marching cubes (scikit-image,
                 opcional) y marching squares para el corte 2D
    metrics   -- topologia (chi, genero, bordes) y error contra la superficie real
    pipeline  -- orquesta todo lo anterior
    viz       -- visor interactivo en Polyscope
    cli       -- linea de comandos (demo / compare / models)
"""

__version__ = "1.0"
