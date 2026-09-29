#!/usr/bin/env python3
"""Demo de reconstruccion de superficies (SGP / CC5513, clase 7).

    python run_recon.py demo                       # blob de genero 1 + visor
    python run_recon.py demo --model bunny --preset hueco
    python run_recon.py compare                    # tabla metodos x regimenes
    python run_recon.py normals                    # error de normales vs k
    python run_recon.py models --fetch all
"""
import os
import sys

# permite ejecutar el script desde cualquier directorio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sr_demo.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
