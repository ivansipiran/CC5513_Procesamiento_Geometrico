#!/usr/bin/env python3
"""Punto de entrada del demo de ICP.

Ejemplos:
    python run_icp.py models
    python run_icp.py demo --model bunny --variant point2plane
    python run_icp.py compare --plot convergencia.png
    python run_icp.py bench-nn --plot escalamiento.png

Este script debe quedar en el mismo directorio que la carpeta del paquete
`icp_demo/` (la que contiene icp.py, neighbors.py, viz.py, etc.).
"""

import os
import sys

# permite ejecutar el script desde cualquier directorio de trabajo
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from icp_demo.cli import main
except ModuleNotFoundError as e:
    if "icp_demo" not in str(e):
        raise
    sys.exit(
        "No se encontro el paquete 'icp_demo'.\n\n"
        f"Se esperaba el directorio: {os.path.join(_HERE, 'icp_demo')}\n"
        "con los archivos __init__.py, cli.py, icp.py, neighbors.py, data.py,\n"
        "geometry.py, scenario.py, viz.py y benchmark.py dentro.\n\n"
        "Descarga el proyecto completo (icp_demo.zip) y descomprimelo conservando\n"
        "la estructura de carpetas:\n\n"
        "    icp_demo/\n"
        "      run_icp.py\n"
        "      icp_demo/\n"
        "        __init__.py  cli.py  icp.py  neighbors.py  data.py\n"
        "        geometry.py  scenario.py  viz.py  benchmark.py\n"
    )

if __name__ == "__main__":
    sys.exit(main())
