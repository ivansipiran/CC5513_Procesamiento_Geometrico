"""
app.py  —  Representaciones 3D (demo interactivo)
=================================================
App de escritorio para MOTIVAR la primera clase del curso
"Procesamiento Geométrico y Análisis de Formas".

Permite:
  * Cargar una malla triangular (una forma generada o tu propio .obj/.ply/.stl).
  * Visualizar simultáneamente sus 3 representaciones:
        - Malla triangular  (superficie)
        - Nube de puntos
        - Voxels (representación volumétrica)
  * Ver, junto a la visualización, el CÓDIGO REAL y la explicación de los
    algoritmos de conversión: malla -> nube  y  malla -> voxels.

Basado en Polyscope (visor pensado para procesamiento geométrico).

Ejecutar:
    python app.py
"""

import inspect
import numpy as np
import polyscope as ps
import polyscope.imgui as psim

import geometry as g


# ======================================================================
# Estado global de la aplicación
# ======================================================================
SHAPES = g.generate_shapes()
SHAPE_NAMES = list(SHAPES.keys())

S = {
    "shape_idx": 0,             # forma generada seleccionada
    "V": None, "F": None,       # malla actual (vértices, caras)
    "model_name": "Toro",

    # nube de puntos
    "n_points": 15000,
    "pc_methods": ["Muestreo por área", "Vértices de la malla"],
    "pc_method_idx": 0,

    # voxels
    "vox_res": 40,
    "vox_solid": True,
    "vox_count": 0,

    # visibilidad
    "show_mesh": True,
    "show_pc": False,
    "show_vox": False,

    # panel de código
    "algo_idx": 0,              # 0 = nube, 1 = voxels
    "path_buf": "",
    "status": "Listo.",
}

# Código fuente de los algoritmos (se muestra tal cual se ejecuta).
SRC_NUBE = inspect.getsource(g.mesh_to_pointcloud)
SRC_VOXEL = (inspect.getsource(g.mesh_to_voxels) + "\n\n" +
             inspect.getsource(g._fill_interior))


# ======================================================================
# Registro / actualización de estructuras en Polyscope
# ======================================================================

def update_mesh():
    """Registra la malla actual como superficie."""
    V, F = S["V"], S["F"]
    m = ps.register_surface_mesh("Malla triangular", V, F, smooth_shade=True)
    m.set_color((0.55, 0.65, 0.85))
    m.set_edge_width(1.0)
    m.set_enabled(S["show_mesh"])


def update_pointcloud():
    """Recalcula y registra la nube de puntos."""
    V, F = S["V"], S["F"]
    method = "area" if S["pc_method_idx"] == 0 else "vertices"
    pts = g.mesh_to_pointcloud(V, F, n_samples=S["n_points"], method=method)
    pc = ps.register_point_cloud("Nube de puntos", pts)
    pc.set_radius(0.0035, relative=False)
    # Color por altura (coordenada Z) para que se lea mejor el volumen.
    pc.add_scalar_quantity("altura", pts[:, 2], enabled=True, cmap="viridis")
    pc.set_enabled(S["show_pc"])


def update_voxels():
    """Recalcula la voxelización y la registra como malla de cubos."""
    V, F = S["V"], S["F"]
    occ, origin, pitch = g.mesh_to_voxels(
        V, F, resolution=S["vox_res"], solid=S["vox_solid"])
    S["vox_count"] = int(occ.sum())
    cv, cf = g.voxels_to_cube_mesh(occ, origin, pitch)
    vox = ps.register_surface_mesh("Voxels", cv, cf, smooth_shade=False)
    vox.set_color((0.95, 0.70, 0.25))
    vox.set_edge_width(1.0)
    vox.set_enabled(S["show_vox"])


def apply_visibility():
    if ps.has_surface_mesh("Malla triangular"):
        ps.get_surface_mesh("Malla triangular").set_enabled(S["show_mesh"])
    if ps.has_point_cloud("Nube de puntos"):
        ps.get_point_cloud("Nube de puntos").set_enabled(S["show_pc"])
    if ps.has_surface_mesh("Voxels"):
        ps.get_surface_mesh("Voxels").set_enabled(S["show_vox"])


def set_model(V, F, name):
    """Fija una nueva malla y regenera las tres representaciones."""
    S["V"], S["F"] = V, F
    S["model_name"] = name
    update_mesh()
    update_pointcloud()
    update_voxels()
    S["status"] = f"Modelo: {name}  |  V={len(V)}  F={len(F)}"


def try_load_path(path):
    """Carga una malla desde una ruta de archivo."""
    try:
        V, F = g.load_mesh(path)
        set_model(V, F, path.split("/")[-1].split("\\")[-1])
    except Exception as e:
        S["status"] = f"Error al cargar: {e}"


def open_file_dialog():
    """Abre un selector de archivos nativo (si tkinter está disponible)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.update()
        path = filedialog.askopenfilename(
            title="Selecciona una malla",
            filetypes=[("Mallas 3D", "*.obj *.ply *.stl *.off *.glb *.gltf"),
                       ("Todos", "*.*")])
        root.destroy()
        if path:
            try_load_path(path)
    except Exception as e:
        S["status"] = ("No se pudo abrir el diálogo (usa el campo de ruta). "
                       f"[{e}]")


# ======================================================================
# Interfaz (se dibuja cada frame dentro del visor de Polyscope)
# ======================================================================

def callback():
    psim.Begin("Representaciones 3D", True)

    psim.TextUnformatted("Curso: Procesamiento Geometrico y Analisis de Formas")
    psim.TextUnformatted(S["status"])
    psim.Separator()

    # ---- Modelo -------------------------------------------------------
    if psim.CollapsingHeader("1. Modelo 3D", psim.ImGuiTreeNodeFlags_DefaultOpen):
        psim.TextUnformatted("Forma generada:")
        psim.SetNextItemWidth(220)
        changed, S["shape_idx"] = psim.Combo("##forma", S["shape_idx"], SHAPE_NAMES)
        psim.SameLine()
        if psim.Button("Cargar forma"):
            name = SHAPE_NAMES[S["shape_idx"]]
            V, F = SHAPES[name]
            set_model(V, F, name)

        psim.Separator()
        psim.TextUnformatted("Tu propia malla (.obj / .ply / .stl):")
        if psim.Button("Examinar archivo..."):
            open_file_dialog()
        psim.SetNextItemWidth(300)
        _, S["path_buf"] = psim.InputText("##ruta", S["path_buf"])
        psim.SameLine()
        if psim.Button("Cargar ruta"):
            if S["path_buf"].strip():
                try_load_path(S["path_buf"].strip())

    # ---- Representaciones visibles -----------------------------------
    if psim.CollapsingHeader("2. Representaciones", psim.ImGuiTreeNodeFlags_DefaultOpen):
        ch, S["show_mesh"] = psim.Checkbox("Malla triangular (superficie)", S["show_mesh"])
        if ch: apply_visibility()
        ch, S["show_pc"] = psim.Checkbox("Nube de puntos", S["show_pc"])
        if ch: apply_visibility()
        ch, S["show_vox"] = psim.Checkbox("Voxels (volumetrica)", S["show_vox"])
        if ch: apply_visibility()

    # ---- Controles de la nube ----------------------------------------
    if psim.CollapsingHeader("3. Malla -> Nube de puntos"):
        psim.SetNextItemWidth(220)
        ch1, S["pc_method_idx"] = psim.Combo("Metodo", S["pc_method_idx"], S["pc_methods"])
        psim.SetNextItemWidth(220)
        ch2, S["n_points"] = psim.SliderInt("N puntos", S["n_points"], 500, 100000)
        if psim.Button("Recalcular nube") or ch1 or ch2:
            update_pointcloud()
            S["show_pc"] = True
            apply_visibility()
        psim.TextUnformatted(f"Puntos actuales: {S['n_points']}")

    # ---- Controles de los voxels -------------------------------------
    if psim.CollapsingHeader("4. Malla -> Voxels"):
        psim.SetNextItemWidth(220)
        ch1, S["vox_res"] = psim.SliderInt("Resolucion", S["vox_res"], 8, 96)
        ch2, S["vox_solid"] = psim.Checkbox("Relleno solido (interior)", S["vox_solid"])
        if psim.Button("Recalcular voxels") or ch1 or ch2:
            update_voxels()
            S["show_vox"] = True
            apply_visibility()
        psim.TextUnformatted(f"Voxels ocupados: {S['vox_count']}")
        psim.TextUnformatted(f"Grilla: {S['vox_res']}^3 "
                             f"(~{S['vox_res']**3:,} celdas)")

    # ---- Panel de algoritmo (código + explicación) -------------------
    if psim.CollapsingHeader("5. Algoritmo de conversion", psim.ImGuiTreeNodeFlags_DefaultOpen):
        ch, S["algo_idx"] = psim.Combo("Ver", S["algo_idx"],
                                       ["Malla -> Nube", "Malla -> Voxels"])
        if S["algo_idx"] == 0:
            psim.TextWrapped(g.EXPLICACION_NUBE)
            src = SRC_NUBE
        else:
            psim.TextWrapped(g.EXPLICACION_VOXEL)
            src = SRC_VOXEL
        psim.Separator()
        psim.TextUnformatted("Codigo que se ejecuta:")
        psim.BeginChild("codigo", (0.0, 260.0), True)
        psim.TextUnformatted(src)
        psim.EndChild()

    psim.End()


# ======================================================================
# Main
# ======================================================================

def main():
    ps.set_program_name("Representaciones 3D — Procesamiento Geométrico")
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")
    ps.init()

    # Modelo inicial: una forma generada.
    name = SHAPE_NAMES[S["shape_idx"]]
    V, F = SHAPES[name]
    set_model(V, F, name)

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
