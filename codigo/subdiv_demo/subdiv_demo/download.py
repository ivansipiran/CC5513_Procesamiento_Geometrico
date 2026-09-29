"""
Descarga robusta (misma cadena que los demos anteriores).

En Windows + conda, ssl.create_default_context() puede fallar leyendo el almacén de
certificados del sistema ([ASN1: NOT_ENOUGH_DATA]) antes de conectarse. Probamos en orden:
    1. contexto con el bundle de certifi
    2. contexto por defecto del sistema
    3. requests (si está instalado)
    4. sin verificar certificado (solo con --insecure)
"""
import shutil
import ssl
import urllib.request


def _urlopen(url, path, ctx):
    req = urllib.request.Request(url, headers={"User-Agent": "subdiv-demo"})
    with urllib.request.urlopen(req, context=ctx, timeout=60) as r, open(path, "wb") as fh:
        shutil.copyfileobj(r, fh)


def fetch(url, path, insecure=False):
    errors = []
    try:
        import certifi
        _urlopen(url, path, ssl.create_default_context(cafile=certifi.where()))
        return
    except Exception as e:
        errors.append(f"certifi: {e!r}")
    try:
        _urlopen(url, path, ssl.create_default_context())
        return
    except Exception as e:
        errors.append(f"sistema: {e!r}")
    try:
        import requests
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as fh:
            fh.write(r.content)
        return
    except Exception as e:
        errors.append(f"requests: {e!r}")
    if insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        _urlopen(url, path, ctx)
        return
    raise RuntimeError("No se pudo descargar " + url + "\n  " + "\n  ".join(errors) +
                       "\nPruebe con --insecure o descargue el archivo a mano en data/")
