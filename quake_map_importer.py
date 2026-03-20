bl_info = {
    "name":        "Quake .map Importer (Valve 220)",
    "author":      "Custom",
    "version":     (1, 0, 0),
    "blender":     (5, 1, 0),
    "location":    "File > Import > Quake Map (.map)  |  3D Viewport > N-Panel > Quake Map",
    "description": "Import Quake .map files (Valve 220 format) with geometry, "
                   "UV mapping, and PNG texture loading. Supports scale presets "
                   "for Unreal Engine 5, Unity, and Godot.",
    "category":    "Import-Export",
    "doc_url":     "",
    "tracker_url": "",
}


import bpy
import bmesh
import math
import re
import hashlib
import os
from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty, BoolProperty, FloatProperty, PointerProperty
from bpy.types import Operator, Panel, PropertyGroup
from mathutils import Vector
from itertools import combinations

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

EPSILON           = 1e-5
PARENT_COLLECTION = "QuakeMap"

# -----------------------------------------------------------------------------
# MATH HELPERS
# -----------------------------------------------------------------------------

def plane_from_three_points(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    # Quake winding is CW from outside — flip cross product for outward normal
    normal = v2.cross(v1)
    if normal.length < EPSILON:
        return None, None
    normal.normalize()
    return normal, normal.dot(p1)


def intersect_three_planes(n1, d1, n2, d2, n3, d3):
    det = (n1.x * (n2.y * n3.z - n2.z * n3.y)
         - n1.y * (n2.x * n3.z - n2.z * n3.x)
         + n1.z * (n2.x * n3.y - n2.y * n3.x))
    if abs(det) < EPSILON:
        return None

    def cramer(col):
        dc   = [d1, d2, d3]
        rows = [list(n1), list(n2), list(n3)]
        for i in range(3):
            rows[i] = rows[i][:]
            rows[i][col] = dc[i]
        return (rows[0][0] * (rows[1][1]*rows[2][2] - rows[1][2]*rows[2][1])
              - rows[0][1] * (rows[1][0]*rows[2][2] - rows[1][2]*rows[2][0])
              + rows[0][2] * (rows[1][0]*rows[2][1] - rows[1][1]*rows[2][0]))

    return Vector((cramer(0)/det, cramer(1)/det, cramer(2)/det))


def point_inside_brush(point, planes, epsilon=EPSILON * 10):
    for normal, distance in planes:
        if normal.dot(point) - distance > epsilon:
            return False
    return True


def compute_brush_vertices(planes):
    verts = []
    for i, j, k in combinations(range(len(planes)), 3):
        n1, d1 = planes[i]
        n2, d2 = planes[j]
        n3, d3 = planes[k]
        pt = intersect_three_planes(n1, d1, n2, d2, n3, d3)
        if pt is None:
            continue
        if point_inside_brush(pt, planes):
            if not any((pt - v).length < EPSILON * 10 for v in verts):
                verts.append(pt)
    return verts


# -----------------------------------------------------------------------------
# VALVE 220 UV
# Quake UV values are in pixel space.  Dividing by image dimensions converts
# them to the 0-1 normalized space that Blender expects.
# If no image is available at build time, we fall back to a 128x128 assumption
# which can be corrected later by re-applying textures.
# -----------------------------------------------------------------------------

DEFAULT_TEX_SIZE = 128  # fallback when image size is unknown


def calc_valve220_uv(vertex, uv_u, uv_v, offset_u, offset_v,
                     scale_u, scale_v, img_w, img_h):
    """
    Compute normalized (0-1) UV for a vertex using Valve 220 axes.

    Valve 220 formula (pixel space):
        u_px = dot(vertex, uv_u) / scale_u + offset_u
        v_px = dot(vertex, uv_v) / scale_v + offset_v

    Blender normalized:
        u = u_px / image_width
        v = v_px / image_height   (V is NOT flipped — Blender handles that)
    """
    if abs(scale_u) < EPSILON: scale_u = 1.0
    if abs(scale_v) < EPSILON: scale_v = 1.0
    u_px = (vertex.dot(uv_u) / scale_u) + offset_u
    v_px = (vertex.dot(uv_v) / scale_v) + offset_v
    return u_px / img_w, v_px / img_h


# -----------------------------------------------------------------------------
# .MAP PARSER
# -----------------------------------------------------------------------------

VALVE220_FACE_RE = re.compile(
    r'\(\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\)'
    r'\s*\(\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\)'
    r'\s*\(\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\)'
    r'\s+(?:"([^"]+)"|(\S+))'
    r'\s*\[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\]'
    r'\s*\[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\]'
    r'\s+([-\d.]+)'
    r'\s+([-\d.]+)\s+([-\d.]+)'
)


def parse_map_file(filepath):
    entities = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    i = 0
    total = len(lines)

    def skip_blank():
        nonlocal i
        while i < total:
            s = lines[i].strip()
            if s == '' or s.startswith('//'):
                i += 1
            else:
                break

    while i < total:
        skip_blank()
        if i >= total:
            break
        if lines[i].strip() != '{':
            i += 1
            continue

        entity = {'properties': {}, 'brushes': []}
        i += 1

        while i < total:
            skip_blank()
            if i >= total:
                break
            line = lines[i].strip()

            if line == '}':
                i += 1
                break

            elif line == '{':
                brush = {'faces': []}
                i += 1
                while i < total:
                    skip_blank()
                    if i >= total:
                        break
                    bl = lines[i].strip()
                    if bl == '}':
                        i += 1
                        break
                    m = VALVE220_FACE_RE.match(bl)
                    if m:
                        g = m.groups()
                        tex = g[9] if g[9] is not None else g[10]
                        brush['faces'].append({
                            'points': [
                                Vector((float(g[0]),  float(g[1]),  float(g[2]))),
                                Vector((float(g[3]),  float(g[4]),  float(g[5]))),
                                Vector((float(g[6]),  float(g[7]),  float(g[8]))),
                            ],
                            'texture':  tex,
                            'uv_u':     Vector((float(g[11]), float(g[12]), float(g[13]))),
                            'offset_u': float(g[14]),
                            'uv_v':     Vector((float(g[15]), float(g[16]), float(g[17]))),
                            'offset_v': float(g[18]),
                            'rotation': float(g[19]),
                            'scale_u':  float(g[20]),
                            'scale_v':  float(g[21]),
                        })
                    i += 1
                entity['brushes'].append(brush)

            elif line.startswith('"'):
                kv = re.findall(r'"([^"]*)"', line)
                if len(kv) >= 2:
                    entity['properties'][kv[0]] = kv[1]
                i += 1
            else:
                i += 1

        entities.append(entity)

    return entities


# -----------------------------------------------------------------------------
# COORDINATE CONVERSION  (Quake -> Blender)
# Quake : X=East, Y=South, Z=Up
# Blender: X=East, Y=North, Z=Up
# Negate Y. Mirrors geometry and reverses winding — compensated by reversing
# vert order when building bmesh faces.
# -----------------------------------------------------------------------------

def q2b(v, scale):
    return Vector((v.x * scale, -v.y * scale, v.z * scale))


# -----------------------------------------------------------------------------
# TEXTURE INDEX  +  MATERIAL / IMAGE HELPERS
# -----------------------------------------------------------------------------

_mat_cache = {}   # { tex_name: bpy.types.Material }


def build_texture_index(folder):
    """Walk folder recursively, return { lowercase_stem: abs_path } for .png."""
    index = {}
    if not folder or not os.path.isdir(folder):
        return index
    for root, _dirs, files in os.walk(folder):
        for fname in files:
            if fname.lower().endswith('.png'):
                stem = os.path.splitext(fname)[0].lower()
                index[stem] = os.path.join(root, fname)
    return index


def load_image(tex_name, tex_index):
    """
    Look up tex_name in tex_index and return (bpy.types.Image, width, height).
    Returns (None, DEFAULT_TEX_SIZE, DEFAULT_TEX_SIZE) if not found.
    """
    stem     = os.path.splitext(os.path.basename(tex_name))[0].lower()
    img_path = tex_index.get(stem)
    if not img_path:
        return None, DEFAULT_TEX_SIZE, DEFAULT_TEX_SIZE

    img_name = os.path.basename(img_path)
    img      = bpy.data.images.get(img_name) or bpy.data.images.load(img_path)
    w, h     = img.size if img.size[0] > 0 else (DEFAULT_TEX_SIZE, DEFAULT_TEX_SIZE)
    return img, w, h


def make_material(tex_name, img):
    """
    Create (or reuse) a Principled BSDF material.
    If img is provided, wire in an Image Texture node.
    """
    mat_name = os.path.basename(tex_name) or tex_name

    if mat_name in _mat_cache:
        return _mat_cache[mat_name]

    mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out  = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    out.location  = (300, 0)
    bsdf.location = (0, 0)
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])

    if img:
        tex_node          = nodes.new('ShaderNodeTexImage')
        tex_node.image    = img
        tex_node.location = (-300, 0)
        links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
        links.new(tex_node.outputs['Alpha'], bsdf.inputs['Alpha'])
        mat.blend_method  = 'CLIP'
    else:
        # Unique viewport colour fallback so brushes are distinguishable
        h = hashlib.md5(tex_name.encode()).digest()
        col = (h[0]/255, h[1]/255, h[2]/255, 1.0)
        bsdf.inputs['Base Color'].default_value = col
        mat.diffuse_color = col

    _mat_cache[mat_name] = mat
    return mat


# -----------------------------------------------------------------------------
# BRUSH MESH BUILDER
# tex_index is passed in so UVs can be normalized against real image dimensions.
# -----------------------------------------------------------------------------

def build_brush_mesh(brush, name, scale, convert_to_mesh, merge_dist, tex_index):
    faces_data = brush['faces']
    if len(faces_data) < 4:
        return None

    # Build planes
    planes = []
    for face in faces_data:
        n, d = plane_from_three_points(*face['points'])
        planes.append((n, d) if n is not None else None)

    valid_planes = [p for p in planes if p is not None]
    if len(valid_planes) < 4:
        return None

    all_verts_q = compute_brush_vertices(valid_planes)
    if len(all_verts_q) < 4:
        return None

    # Pre-resolve images and build material slot list for this brush
    # tex_name -> (material, img_w, img_h, slot_index)
    tex_slots = {}
    mat_list  = []
    for face_info in faces_data:
        tex = face_info['texture']
        if tex not in tex_slots:
            img, w, h = load_image(tex, tex_index)
            mat       = make_material(tex, img)
            slot_idx  = len(mat_list)
            mat_list.append(mat)
            tex_slots[tex] = (mat, w, h, slot_idx)

    bm       = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")
    vert_map = {}

    def get_bm_vert(qv):
        key = (round(qv.x, 4), round(qv.y, 4), round(qv.z, 4))
        if key not in vert_map:
            vert_map[key] = bm.verts.new(q2b(qv, scale))
        return vert_map[key]

    added_faces = 0

    for plane_data, face_info in zip(planes, faces_data):
        if plane_data is None:
            continue

        normal, dist = plane_data
        tex          = face_info['texture']
        _, img_w, img_h, slot_idx = tex_slots[tex]

        face_verts_q = [v for v in all_verts_q
                        if abs(normal.dot(v) - dist) < EPSILON * 10]
        if len(face_verts_q) < 3:
            continue

        # Sort into winding order around centroid on the plane
        centroid = sum(face_verts_q, Vector()) / len(face_verts_q)
        ref      = (face_verts_q[0] - centroid).normalized()
        if ref.length < EPSILON:
            continue
        tangent = normal.cross(ref).normalized()
        face_verts_q.sort(
            key=lambda v: math.atan2((v - centroid).dot(tangent),
                                     (v - centroid).dot(ref))
        )

        # Reverse winding — compensates for Y-negation in q2b()
        bm.verts.ensure_lookup_table()
        bm_verts = [get_bm_vert(v) for v in reversed(face_verts_q)]
        bm.verts.ensure_lookup_table()

        try:
            bm_face = bm.faces.new(bm_verts)
        except ValueError:
            continue

        # Assign correct material slot for this face
        bm_face.material_index = slot_idx

        # Compute normalized UVs using real image dimensions
        for loop, qv in zip(bm_face.loops, reversed(face_verts_q)):
            u, v = calc_valve220_uv(
                qv,
                face_info['uv_u'],     face_info['uv_v'],
                face_info['offset_u'], face_info['offset_v'],
                face_info['scale_u'],  face_info['scale_v'],
                img_w, img_h,
            )
            loop[uv_layer].uv = (u, v)

        added_faces += 1

    if added_faces == 0:
        bm.free()
        return None

    if convert_to_mesh:
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_dist)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)

    # Attach materials in slot order
    for mat in mat_list:
        obj.data.materials.append(mat)

    return obj


# -----------------------------------------------------------------------------
# CORE IMPORT LOGIC
# -----------------------------------------------------------------------------

def import_quake_map(filepath, scale, convert_to_mesh, merge_dist, tex_index=None):
    if tex_index is None:
        tex_index = {}

    print(f"\n{'='*60}")
    print(f"Quake .map Importer  --  Valve 220")
    print(f"File  : {filepath}")
    print(f"Scale : {scale}  |  Mode: {'Mesh' if convert_to_mesh else 'Hull'}")
    print(f"{'='*60}")

    _mat_cache.clear()

    entities = parse_map_file(filepath)
    total_parsed_brushes = sum(len(e.get('brushes', [])) for e in entities)
    total_parsed_faces   = sum(len(b['faces'])
                               for e in entities for b in e.get('brushes', []))
    print(f"Parsed  : {len(entities)} entities | "
          f"{total_parsed_brushes} brushes | {total_parsed_faces} faces")

    if total_parsed_brushes == 0:
        print("[ERROR] No brushes found — verify the file is Valve 220 format.")
        return

    # Wipe old collection
    if PARENT_COLLECTION in bpy.data.collections:
        old = bpy.data.collections[PARENT_COLLECTION]
        for obj in list(old.all_objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        for scene in bpy.data.scenes:
            if old.name in scene.collection.children:
                scene.collection.children.unlink(old)
        bpy.data.collections.remove(old)

    root_col = bpy.data.collections.new(PARENT_COLLECTION)
    bpy.context.scene.collection.children.link(root_col)
    print(f"Created : collection '{root_col.name}'")

    def parse_origin(p):
        if 'origin' in p:
            parts = p['origin'].split()
            if len(parts) == 3:
                return q2b(Vector((float(parts[0]),
                                   float(parts[1]),
                                   float(parts[2]))), scale)
        return Vector((0.0, 0.0, 0.0))

    total_brushes = 0
    total_skipped = 0

    for ent_idx, entity in enumerate(entities):
        props     = entity.get('properties', {})
        classname = props.get('classname', f'entity_{ent_idx}')
        brushes   = entity.get('brushes', [])
        is_world  = (classname == 'worldspawn')

        if not brushes:
            empty = bpy.data.objects.new(f"{classname}_{ent_idx:03d}", None)
            empty.empty_display_type = 'ARROWS'
            empty.empty_display_size = 0.25
            empty.location = parse_origin(props)
            for k, v in props.items():
                empty[k] = v
            root_col.objects.link(empty)
            continue

        if not is_world:
            parent_empty = bpy.data.objects.new(f"{classname}_{ent_idx:03d}", None)
            parent_empty.empty_display_type = 'CUBE'
            parent_empty.empty_display_size = 0.1
            parent_empty.location = parse_origin(props)
            for k, v in props.items():
                parent_empty[k] = v
            root_col.objects.link(parent_empty)
        else:
            parent_empty = None

        for b_idx, brush in enumerate(brushes):
            brush_name = (f"brush_{ent_idx:03d}_{b_idx:03d}" if is_world
                          else f"{classname}_{ent_idx:03d}_brush_{b_idx:03d}")

            obj = build_brush_mesh(
                brush, brush_name, scale, convert_to_mesh, merge_dist, tex_index
            )
            if obj is None:
                total_skipped += 1
                continue

            root_col.objects.link(obj)
            if parent_empty is not None:
                obj.parent = parent_empty
            total_brushes += 1

        print(f"  [{ent_idx:03d}] {classname:30s}  brushes: {len(brushes)}")

    print(f"\n{'='*60}")
    print(f"Objects in '{PARENT_COLLECTION}' : {len(list(root_col.all_objects))}")
    print(f"Brushes imported               : {total_brushes}")
    print(f"Brushes skipped                : {total_skipped}")
    print(f"{'='*60}")

    bpy.ops.object.select_all(action='DESELECT')
    for obj in root_col.all_objects:
        obj.select_set(True)

    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                region = next((r for r in area.regions if r.type == 'WINDOW'), None)
                if region:
                    with bpy.context.temp_override(window=window, area=area, region=region):
                        bpy.ops.view3d.view_selected(use_all_regions=False)
                break


# -----------------------------------------------------------------------------
# N-PANEL SETTINGS
# -----------------------------------------------------------------------------

class QuakeMapSettings(PropertyGroup):
    texture_folder: StringProperty(
        name        = "Texture Folder",
        description = "Root folder searched recursively for .png texture files",
        default     = "",
        subtype     = 'DIR_PATH',
    )


# -----------------------------------------------------------------------------
# RE-APPLY TEXTURES OPERATOR
# Rebuilds materials with correct UVs using newly chosen texture folder.
# Because UV normalization needs image dimensions, this triggers a full
# re-import of the last .map file with the new tex_index.
# -----------------------------------------------------------------------------

class QMAP_OT_load_textures(Operator):
    """Apply PNG textures to all imported QuakeMap brush objects"""
    bl_idname  = "qmap.load_textures"
    bl_label   = "Apply Textures"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        folder = context.scene.qmap_settings.texture_folder.strip()

        if not folder:
            self.report({'WARNING'}, "No texture folder set.")
            return {'CANCELLED'}

        if not os.path.isdir(folder):
            self.report({'ERROR'}, f"Folder not found: {folder}")
            return {'CANCELLED'}

        tex_index = build_texture_index(folder)
        print(f"\n[QuakeMap] Texture folder  : {folder}")
        print(f"[QuakeMap] PNGs indexed    : {len(tex_index)}")

        col = bpy.data.collections.get(PARENT_COLLECTION)
        if col is None:
            self.report({'WARNING'},
                        f"No '{PARENT_COLLECTION}' collection — import a map first.")
            return {'CANCELLED'}

        applied = 0
        missing = 0

        for obj in col.all_objects:
            if obj.type != 'MESH':
                continue

            for slot in obj.material_slots:
                mat = slot.material
                if mat is None:
                    continue

                # Derive tex_name from material name for the index lookup
                tex_name = mat.name
                stem     = os.path.splitext(os.path.basename(tex_name))[0].lower()
                img_path = tex_index.get(stem)

                if not img_path:
                    missing += 1
                    continue

                # Load / reuse image
                img_name = os.path.basename(img_path)
                img      = (bpy.data.images.get(img_name)
                            or bpy.data.images.load(img_path))
                img_w, img_h = img.size if img.size[0] > 0 else (DEFAULT_TEX_SIZE, DEFAULT_TEX_SIZE)

                # Wire image into the material's node graph
                mat.use_nodes = True
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links

                bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
                if bsdf is None:
                    missing += 1
                    continue

                for n in [n for n in nodes if n.type == 'TEX_IMAGE']:
                    nodes.remove(n)

                tex_node          = nodes.new('ShaderNodeTexImage')
                tex_node.image    = img
                tex_node.location = (-300, 0)
                links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
                links.new(tex_node.outputs['Alpha'], bsdf.inputs['Alpha'])
                mat.blend_method  = 'CLIP'

                # Recompute UVs on the mesh using real image dimensions.
                # Walk every face of this mesh, check its material slot,
                # re-project UVs from the stored face plane data.
                # NOTE: UV reprojection after-the-fact requires the original
                # Quake-space vert positions, which we no longer have post-import.
                # So we rescale existing UVs from DEFAULT_TEX_SIZE to real size.
                uv_map = obj.data.uv_layers.active
                if uv_map:
                    slot_idx = obj.material_slots.find(mat.name)
                    scale_u  = DEFAULT_TEX_SIZE / img_w
                    scale_v  = DEFAULT_TEX_SIZE / img_h
                    for poly in obj.data.polygons:
                        if poly.material_index == slot_idx:
                            for loop_idx in poly.loop_indices:
                                uv = uv_map.data[loop_idx].uv
                                uv_map.data[loop_idx].uv = (uv.x * scale_u,
                                                             uv.y * scale_v)

                applied += 1
                print(f"  [TEX] '{mat.name}'  ->  {img_name}  ({img_w}x{img_h})")

        self.report({'INFO'}, f"Applied: {applied}  |  Not found: {missing}")
        print(f"[QuakeMap] Applied: {applied}  |  Missing: {missing}")
        return {'FINISHED'}


# -----------------------------------------------------------------------------
# N-PANEL
# -----------------------------------------------------------------------------

class QMAP_PT_panel(Panel):
    bl_label       = "Quake Map"
    bl_idname      = "QMAP_PT_panel"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = "Quake Map"

    def draw(self, context):
        layout   = self.layout
        settings = context.scene.qmap_settings

        layout.label(text="Post-Import Textures", icon='TEXTURE')
        layout.separator()
        col = layout.column(align=True)
        col.prop(settings, "texture_folder", text="Folder")
        col.separator()
        col.operator("qmap.load_textures", icon='MATERIAL')


# -----------------------------------------------------------------------------
# IMPORT OPERATOR   (File -> Import -> Quake Map (.map))
# -----------------------------------------------------------------------------

# Scale presets: Quake 1 unit = 1 inch (2.54 cm = 0.0254 m)
#   Unreal Engine 5 : 1 unit = 1 cm  ->  scale = 2.54
#   Unity / Godot   : 1 unit = 1 m   ->  scale = 0.0254
#   Blender (metres): common approx  ->  scale = 0.03125  (kept as Custom default)
SCALE_PRESETS = [
    ('UNREAL',  "Unreal Engine 5",  "1 Quake unit = 2.54 cm  (UE5 units)",  2.54),
    ('UNITY',   "Unity",            "1 Quake unit = 0.0254 m (Unity units)", 0.0254),
    ('GODOT',   "Godot",            "1 Quake unit = 0.0254 m (Godot units)", 0.0254),
    ('CUSTOM',  "Custom",           "Enter a scale value manually",          0.03125),
]
SCALE_PRESET_ITEMS = [(p[0], p[1], p[2]) for p in SCALE_PRESETS]
SCALE_PRESET_VALUES = {p[0]: p[3] for p in SCALE_PRESETS}


class IMPORT_OT_quake_map(Operator, ImportHelper):
    """Import a Quake .map file (Valve 220 format)"""
    bl_idname  = "import_scene.quake_map"
    bl_label   = "Import Quake Map"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".map"
    filter_glob: StringProperty(default="*.map", options={'HIDDEN'})

    scale_preset: bpy.props.EnumProperty(
        name        = "Engine",
        description = "Target engine — sets the scale automatically",
        items       = SCALE_PRESET_ITEMS,
        default     = 'UNREAL',
    )

    import_scale: FloatProperty(
        name        = "Custom Scale",
        description = "Manual scale override (only used when Engine is set to Custom). Multiplies every Quake coordinate by this value",
        default     = 0.03125,
        min=0.00001, max=1000.0, precision=5,
    )

    convert_to_mesh: BoolProperty(
        name        = "Convert to Mesh",
        description = "Merge verts and recalc normals. Disable for raw convex hull.",
        default     = True,
    )

    merge_distance: FloatProperty(
        name        = "Merge Distance",
        description = "Vertex merge threshold (in target-engine units)",
        default     = 0.0001,
        min=0.0, max=1.0, precision=6,
    )

    def get_scale(self):
        if self.scale_preset == 'CUSTOM':
            return self.import_scale
        return SCALE_PRESET_VALUES[self.scale_preset]

    def execute(self, context):
        scale = self.get_scale()

        # Pick up the texture folder from the N-panel if already set
        tex_folder = context.scene.qmap_settings.texture_folder.strip()
        tex_index  = build_texture_index(tex_folder) if tex_folder else {}
        if tex_index:
            print(f"[QuakeMap] Texture folder: {tex_folder} ({len(tex_index)} PNGs)")

        import_quake_map(
            filepath        = self.filepath,
            scale           = scale,
            convert_to_mesh = self.convert_to_mesh,
            merge_dist      = self.merge_distance,
            tex_index       = tex_index,
        )
        self.report({'INFO'}, f"Quake .map imported at scale {scale:.6g}.")
        return {'FINISHED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True

        # ── Engine / Scale ────────────────────────────────────────────────────
        layout.label(text="Target Engine")
        layout.prop(self, "scale_preset", text="Engine")

        # Show the resolved scale value as read-only info
        scale = self.get_scale()
        if self.scale_preset != 'CUSTOM':
            row = layout.row()
            row.enabled = False
            row.label(text=f"Scale:  {scale:.6g}  (1 Quake unit → {scale:.6g} engine units)")
        else:
            layout.prop(self, "import_scale", text="Scale")

        # ── Geometry ──────────────────────────────────────────────────────────
        layout.separator()
        layout.label(text="Geometry")
        layout.prop(self, "convert_to_mesh")
        col = layout.column()
        col.enabled = self.convert_to_mesh
        col.prop(self, "merge_distance")


# -----------------------------------------------------------------------------
# MENU + REGISTER
# -----------------------------------------------------------------------------

def menu_func_import(self, context):
    self.layout.operator(IMPORT_OT_quake_map.bl_idname, text="Quake Map (.map)")


_classes = [
    QuakeMapSettings,
    QMAP_OT_load_textures,
    QMAP_PT_panel,
    IMPORT_OT_quake_map,
]


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.qmap_settings = PointerProperty(type=QuakeMapSettings)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    print("[QuakeMapImporter] Registered  ->  File > Import > Quake Map (.map)")
    print("[QuakeMapImporter] N-Panel     ->  3D Viewport > N > 'Quake Map' tab")


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    if hasattr(bpy.types.Scene, 'qmap_settings'):
        del bpy.types.Scene.qmap_settings
    for cls in reversed(_classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
    print("[QuakeMapImporter] Unregistered.")


if __name__ == "__main__":
    try:
        unregister()
    except Exception:
        pass
    register()
