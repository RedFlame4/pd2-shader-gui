"""Parser/serializer for Diesel engine .shaders packages (PAYDAY 2).

Port of payday2-shader-tool (ShaderPackage.kt, PersistentObject.kt,
ByteBufferUtils.kt). Saving an unmodified package is byte-identical to the
input, except that the leading file-size header is recomputed rather than
preserved (see ShaderPackage), so a file that arrived with a stale size is
corrected on save.
"""

import math
import struct

TYPE_SHADER = 0x7F3552D1
TYPE_SHADER_PASS = 0x214B1AAF
TYPE_SHADER_LIBRARY = 0x12812C1A


class Reader:
    def __init__(self, data, pos=0, end=None):
        self.data = data
        self.pos = pos
        self.end = len(data) if end is None else end

    def _take(self, fmt, size):
        if self.pos + size > self.end:
            raise ValueError("Read past end of object")
        v = struct.unpack_from(fmt, self.data, self.pos)[0]
        self.pos += size
        return v

    def i32(self):
        return self._take("<i", 4)

    def u32(self):
        return self._take("<I", 4)

    def u64(self):
        return self._take("<Q", 8)

    def u8(self):
        if self.pos >= self.end:
            raise ValueError("Read past end of object")
        b = self.data[self.pos]
        self.pos += 1
        return b

    def cstring(self):
        zero = self.data.index(b"\0", self.pos, self.end)
        s = self.data[self.pos:zero].decode("utf-8")
        self.pos = zero + 1
        return s

    def len_array(self):
        n = self.u32()
        if self.pos + n > self.end:
            raise ValueError("Read past end of object")
        b = bytes(self.data[self.pos:self.pos + n])
        self.pos += n
        return b

    def remaining(self):
        return self.end - self.pos


class Writer:
    def __init__(self):
        self.buf = bytearray()

    def i32(self, v):
        self.buf += struct.pack("<i", v)

    def u32(self, v):
        self.buf += struct.pack("<I", v)

    def u64(self, v):
        self.buf += struct.pack("<Q", v)

    def u8(self, v):
        self.buf.append(v & 0xFF)

    def cstring(self, s):
        self.buf += s.encode("utf-8") + b"\0"

    def len_array(self, b):
        self.u32(len(b))
        self.buf += b


class StateVar:
    """id:u32, flag:u8, then u32 value (flag==0) or u64 Idstring hash."""

    def __init__(self, var_id, flag, val4=None, val8=None):
        self.id = var_id
        self.flag = flag
        self.val4 = val4
        self.val8 = val8

    @classmethod
    def load(cls, r):
        var_id = r.i32()
        flag = r.u8()
        if flag == 0:
            return cls(var_id, flag, val4=r.u32())
        return cls(var_id, flag, val8=r.u64())

    def save(self, w):
        w.i32(self.id)
        w.u8(self.flag)
        if self.flag == 0:
            w.u32(self.val4 & 0xFFFFFFFF)
        else:
            w.u64(self.val8)


class TextureBlock:
    """A per-sampler block of state vars.

    In D3D9 passes `tex_id` is a small "unknown int" (a sampler-stage
    register index); sampler names live in the bytecode's CTAB constant
    table instead. In D3D11 passes there is no CTAB to recover names from
    (bytecode is DXBC/SM4-5), so `tex_id` is instead the full 8-byte Idstring
    hash of the sampler/texture variable name (resolvable via the hashlist,
    same as render-template/mode names).
    """

    def __init__(self, tex_id, svars):
        self.tex_id = tex_id
        self.vars = svars


def _blob_kind(blob):
    """Classify a bytecode blob by its magic:

    "d3d11" for DXBC (D3D10/11, SM4-5), "d3d9" for a D3D9 SM1-3 version token,
    or None when empty/unrecognised. The DXBC magic ('DXBC' = 0x43425844) is
    the authoritative D3D11-vs-D3D9 signal, since both layouts share the same
    outer container and a D3D11 pass can otherwise parse cleanly as D3D9.
    """
    if len(blob) >= 4:
        if blob[:4] == b"DXBC":
            return "d3d11"
        tok = struct.unpack_from("<I", blob, 0)[0]
        if (tok >> 16) in (0xFFFE, 0xFFFF):
            return "d3d9"
    return None


def _looks_like_shader_blob(blob):
    """True if `blob` starts like D3D9 SM1-3 bytecode, DXBC (D3D10/11), or is empty."""
    return len(blob) == 0 or _blob_kind(blob) is not None


class ObjShaderPass:
    """`layout` is "d3d9" (raw D3DRENDERSTATETYPE ids, int texture-block
    header) or "d3d11" (Diesel's own compact D3D11 state-var ids, Idstring
    texture-block header); detected on load from the texture-block/bytecode
    shape since both share the same outer container format."""

    def __init__(self, hdr):
        self.hdr = hdr
        self.state_vars = []
        self.textures = []
        self.vertex_shader = b""
        self.fragment_shader = b""
        self.layout = "d3d9"

    def load(self, r, ref_map):
        self.state_vars = [StateVar.load(r) for _ in range(r.u32())]

        # Both layouts share the same outer container, so a D3D11 pass can
        # parse cleanly as D3D9 (tex_id read as i32 instead of u64) and vice
        # versa. Collect every layout that parses to the end with valid-looking
        # bytecode, then let the DXBC magic decide (see _blob_kind).
        tex_start = r.pos
        candidates = []
        for layout in ("d3d9", "d3d11"):
            r.pos = tex_start
            try:
                textures = []
                for _ in range(r.u32()):
                    tex_id = r.u64() if layout == "d3d11" else r.i32()
                    svars = [StateVar.load(r) for _ in range(r.u32())]
                    textures.append(TextureBlock(tex_id, svars))
                vertex_shader = r.len_array()
                fragment_shader = r.len_array()
            except ValueError:
                continue
            if r.pos == r.end and _looks_like_shader_blob(vertex_shader) \
                    and _looks_like_shader_blob(fragment_shader):
                candidates.append((layout, textures, vertex_shader,
                                   fragment_shader))

        chosen = self._pick_candidate(candidates)
        if chosen is None:
            raise ValueError("Shader pass refId=%d: unrecognised texture-block "
                             "layout (not D3D9 or D3D11)" % self.hdr.ref_id)

        self.layout, self.textures, self.vertex_shader, self.fragment_shader = \
            chosen
        # Trying each layout leaves the reader wherever the last attempt
        # stopped (mid-object if it raised); every accepted candidate consumed
        # the object exactly, so rewind to the chosen parse's end.
        r.pos = r.end

    @staticmethod
    def _pick_candidate(candidates):
        """Choose among successfully-parsed layouts. The bytecode's DXBC/SM
        magic is authoritative: prefer a candidate whose blob magic matches its
        layout and never one it contradicts (e.g. DXBC bytecode under a D3D9
        parse). Falls back to the first clean parse when the blobs are empty
        and carry no magic."""
        def kinds(cand):
            _, _, vs, fs = cand
            return {k for k in (_blob_kind(vs), _blob_kind(fs)) if k is not None}

        # A candidate whose layout is positively confirmed by the bytecode magic.
        for cand in candidates:
            if kinds(cand) == {cand[0]}:
                return cand
        # Otherwise, the first candidate not contradicted by its bytecode magic.
        for cand in candidates:
            if cand[0] in kinds(cand) or not kinds(cand):
                return cand
        return candidates[0] if candidates else None

    def save(self, w):
        w.u32(len(self.state_vars))
        for sv in self.state_vars:
            sv.save(w)

        w.u32(len(self.textures))
        for block in self.textures:
            if self.layout == "d3d11":
                w.u64(block.tex_id)
            else:
                w.i32(block.tex_id)
            w.u32(len(block.vars))
            for sv in block.vars:
                sv.save(w)

        w.len_array(self.vertex_shader)
        w.len_array(self.fragment_shader)


class ObjShader:
    """Maps mode Idstrings to pass lists. Entry order is preserved on save."""

    def __init__(self, hdr):
        self.hdr = hdr
        self.shader_packs = []  # list of (hash, [ObjShaderPass]) in file order

    def load(self, r, ref_map):
        self.shader_packs = []
        for _ in range(r.u32()):
            h = r.u64()
            passes = [ref_map[r.i32()] for _ in range(r.u32())]
            self.shader_packs.append((h, passes))

    def save(self, w):
        w.u32(len(self.shader_packs))
        for h, passes in self.shader_packs:
            w.u64(h)
            w.u32(len(passes))
            for p in passes:
                w.i32(p.hdr.ref_id)


class ObjShaderLibrary:
    """Maps render-template Idstrings to ObjShaders. Saved sorted by unsigned hash."""

    def __init__(self, hdr):
        self.hdr = hdr
        self.render_templates = {}  # hash -> ObjShader

    def load(self, r, ref_map):
        self.render_templates = {}
        for _ in range(r.u32()):
            h = r.u64()
            self.render_templates[h] = ref_map[r.i32()]

    def save(self, w):
        w.u32(len(self.render_templates))
        for h in sorted(self.render_templates):
            w.u64(h)
            w.i32(self.render_templates[h].hdr.ref_id)


class ObjectHeader:
    def __init__(self, obj_type, ref_id, length, pos):
        self.type = obj_type
        self.ref_id = ref_id
        self.len = length
        self.pos = pos

    def build(self):
        cls = {
            TYPE_SHADER: ObjShader,
            TYPE_SHADER_PASS: ObjShaderPass,
            TYPE_SHADER_LIBRARY: ObjShaderLibrary,
        }.get(self.type)
        if cls is None:
            raise ValueError("Unknown object type 0x%08X" % self.type)
        return cls(self)


class ShaderPackage:
    """Two container headers exist; the object table that follows is identical
    in both, so only the preamble is version-dependent.

    Legacy files open with a -1 marker followed by a u32 holding the total file
    size (the prefix counts towards its own size). Newer files (observed in
    base.d3d11.shaders) open with a 'DODB' magic and a u32 container version
    (1) before that same size word; `dodb_version` records which form was
    found (None for legacy) so save() reproduces it. `has_size_header` records
    whether a size word was present at all, and the size itself is recomputed
    on save.

    Confirmed against an unmodified shaders file, where it matched
    the file length exactly. Files that have been through an editing
    tool carry a stale value, since both the original payday2-shader-tool
    and earlier versions of this code wrote it back verbatim, so a mismatch
    on load (`stale_size_header`) means the file was written by such a tool -
    or, less often, that it is truncated.
    """

    MAGIC_DODB = b"DODB"

    def __init__(self):
        self.objects = []
        self.has_size_header = True
        self.dodb_version = None     # None = legacy -1-marker container
        self.stored_size = None      # header as found on load; diagnostic only
        self.stale_size_header = False

    def load(self, data):
        r = Reader(data)
        self.dodb_version = None
        self.stored_size = None
        if data[:4] == self.MAGIC_DODB:
            r.pos = 4
            self.dodb_version = r.u32()
            if self.dodb_version != 1:
                raise ValueError("Unsupported DODB container version %d"
                                 % self.dodb_version)
            self.has_size_header = True
            self.stored_size = r.u32()
            count = r.i32()
        else:
            count = r.i32()
            self.has_size_header = count == -1
            if self.has_size_header:
                self.stored_size = r.u32()
                count = r.i32()
        self.stale_size_header = (self.has_size_header
                                  and self.stored_size != len(data))

        headers = []
        for _ in range(count):
            obj_type = r.u32()
            ref_id = r.i32()
            length = r.u32()
            headers.append(ObjectHeader(obj_type, ref_id, length, r.pos))
            r.pos += length

        self.objects = [h.build() for h in headers]
        ref_map = {obj.hdr.ref_id: obj for obj in self.objects}

        for obj in self.objects:
            inner = Reader(data, obj.hdr.pos, obj.hdr.pos + obj.hdr.len)
            obj.load(inner, ref_map)
            if inner.pos != inner.end:
                raise ValueError(
                    "Object refId=%d: %d unread bytes" % (obj.hdr.ref_id, inner.end - inner.pos))

    def size_header_offset(self):
        """Byte offset of the u32 file-size word, or None if the file has no
        size header. It follows the container preamble, so it sits at 4 in a
        legacy file and at 8 in a DODB one."""
        if not self.has_size_header:
            return None
        return 8 if self.dodb_version is not None else 4

    def save(self):
        w = Writer()
        size_at = self.size_header_offset()
        if self.dodb_version is not None:
            w.buf += self.MAGIC_DODB
            w.u32(self.dodb_version)
            w.u32(0)  # file size placeholder, filled in below
        elif self.has_size_header:
            w.i32(-1)
            w.u32(0)  # file size placeholder, filled in below

        w.u32(len(self.objects))
        for obj in self.objects:
            item = Writer()
            obj.save(item)
            w.u32(obj.hdr.type)
            w.i32(obj.hdr.ref_id)
            w.u32(len(item.buf))
            w.buf += item.buf

        if size_at is not None:
            # The file size, now that it is known; it sits straight after the
            # container preamble and counts the whole file.
            struct.pack_into("<I", w.buf, size_at, len(w.buf))
        return bytes(w.buf)

    def find_library(self):
        libs = [o for o in self.objects if isinstance(o, ObjShaderLibrary)]
        if len(libs) != 1:
            raise ValueError("Expected exactly 1 shader library, found %d" % len(libs))
        return libs[0]


# --- State variable metadata (port of SVType.kt / SVTypeTex.kt / ArgType.kt) ---

class ArgType:
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    ENUM = "enum"
    BITMASK = "bitmask"


MASK_COLOUR = ["Red", "Green", "Blue", "Alpha"]


class SVDef:
    def __init__(self, name, arg_type, values=None):
        self.name = name
        self.arg_type = arg_type
        self.values = values

    def decode(self, sv):
        """Human-readable value for a StateVar, or None if not representable."""
        if sv.val4 is None:
            return None
        v = sv.val4
        if self.arg_type == ArgType.BOOL:
            return "true" if v != 0 else "false"
        if self.arg_type == ArgType.INT:
            return str(v)
        if self.arg_type == ArgType.FLOAT:
            # Non-finite bit patterns (e.g. 0xffffffff -> nan) aren't editable
            # as floats and would lose their exact bits on re-commit; let the
            # caller fall back to a raw hex view that round-trips losslessly.
            f = struct.unpack("<f", struct.pack("<I", v))[0]
            return repr(f) if math.isfinite(f) else None
        if self.arg_type == ArgType.ENUM:
            if 0 <= v < len(self.values) and self.values[v] is not None:
                return self.values[v]
            return "<invalid enum %d>" % v
        if self.arg_type == ArgType.BITMASK:
            found = [n for i, n in enumerate(self.values) if v & (1 << i)]
            extra = v & ~((1 << len(self.values)) - 1)
            if extra:
                found.append("<unknown bits 0x%x>" % extra)
            return ", ".join(found) if found else "(none)"
        return None

    def encode(self, text):
        """Parse a human-readable value back to a raw u32, raising ValueError on bad input."""
        text = text.strip()
        if self.arg_type == ArgType.BOOL:
            if text.lower() in ("true", "1", "yes", "on"):
                return 1
            if text.lower() in ("false", "0", "no", "off"):
                return 0
            raise ValueError("Expected true/false")
        if self.arg_type == ArgType.INT:
            return int(text, 0) & 0xFFFFFFFF
        if self.arg_type == ArgType.FLOAT:
            return struct.unpack("<I", struct.pack("<f", float(text)))[0]
        if self.arg_type == ArgType.ENUM:
            for i, name in enumerate(self.values):
                if name is not None and name.lower() == text.lower():
                    return i
            return int(text, 0)  # allow raw index too
        if self.arg_type == ArgType.BITMASK:
            if not text or text == "(none)":
                return 0
            bits = 0
            for part in text.split(","):
                part = part.strip()
                for i, name in enumerate(self.values):
                    if name.lower() == part.lower():
                        bits |= 1 << i
                        break
                else:
                    raise ValueError("Unknown flag %r" % part)
            return bits
        raise ValueError("Cannot encode this type")


# D3D-layout files (Windows build): state var IDs are D3DRENDERSTATETYPE
# values and texture var IDs are D3DSAMPLERSTATETYPE values (verified
# against deferred_lighting.d3d9.shaders).

ENUM_D3DCMP = [None, "Never", "Less", "Equal", "Less or Equal", "Greater",
               "Not Equal", "Greater or Equal", "Always"]
ENUM_D3DSTENCILOP = [None, "Keep", "Zero", "Replace", "Increment (sat)",
                     "Decrement (sat)", "Invert", "Increment", "Decrement"]
ENUM_D3DBLEND = [None, "Zero", "One", "Src Colour", "Inv Src Colour",
                 "Src Alpha", "Inv Src Alpha", "Dest Alpha", "Inv Dest Alpha",
                 "Dest Colour", "Inv Dest Colour", "Src Alpha Sat",
                 "Both Src Alpha", "Both Inv Src Alpha", "Blend Factor",
                 "Inv Blend Factor", "Src Colour 2", "Inv Src Colour 2"]
ENUM_D3DBLENDOP = [None, "Add", "Subtract", "Reverse-Subtract", "Min", "Max"]
ENUM_D3DCULL = [None, "None", "Clockwise", "Counter-Clockwise"]
ENUM_D3DZB = ["Disabled", "Enabled (Z)", "Enabled (W)"]
ENUM_D3DFILL = [None, "Point", "Wireframe", "Solid"]
ENUM_D3DSHADE = [None, "Flat", "Gouraud", "Phong"]
ENUM_D3DADDRESS = [None, "Wrap", "Mirror", "Clamp", "Border", "Mirror Once"]
ENUM_D3DFILTER = ["None", "Point", "Linear", "Anisotropic", None, None,
                  "Pyramidal Quad", "Gaussian Quad"]
ENUM_D3DFOGMODE = ["None", "Exp", "Exp2", "Linear"]
ENUM_D3DMATSOURCE = ["Material", "Colour 1 (diffuse)", "Colour 2 (specular)"]
ENUM_D3DDEGREE = [None, "Linear", "Quadratic", "Cubic", None, "Quintic"]
ENUM_D3DPATCHEDGE = ["Discrete", "Continuous"]
ENUM_D3DDMT = ["Enable", "Disable"]
ENUM_D3DVERTEXBLEND = ["Disable", "1 Weight", "2 Weights", "3 Weights"]
MASK_WRAPCOORD = ["U (coord 0)", "V (coord 1)", "W (coord 2)", "Coord 3"]
MASK_CLIPPLANES = ["Plane 0", "Plane 1", "Plane 2", "Plane 3", "Plane 4", "Plane 5"]

SV_TYPES_D3D = {
    7: SVDef("ZEnable", ArgType.ENUM, ENUM_D3DZB),
    8: SVDef("FillMode", ArgType.ENUM, ENUM_D3DFILL),
    9: SVDef("ShadeMode", ArgType.ENUM, ENUM_D3DSHADE),
    14: SVDef("ZWriteEnable", ArgType.BOOL),
    15: SVDef("AlphaTestEnable", ArgType.BOOL),
    16: SVDef("LastPixel", ArgType.BOOL),
    19: SVDef("SrcBlend", ArgType.ENUM, ENUM_D3DBLEND),
    20: SVDef("DestBlend", ArgType.ENUM, ENUM_D3DBLEND),
    22: SVDef("CullMode", ArgType.ENUM, ENUM_D3DCULL),
    23: SVDef("ZFunc", ArgType.ENUM, ENUM_D3DCMP),
    24: SVDef("AlphaRef", ArgType.INT),
    25: SVDef("AlphaFunc", ArgType.ENUM, ENUM_D3DCMP),
    26: SVDef("DitherEnable", ArgType.BOOL),
    27: SVDef("AlphaBlendEnable", ArgType.BOOL),
    28: SVDef("FogEnable", ArgType.BOOL),
    29: SVDef("SpecularEnable", ArgType.BOOL),
    34: SVDef("FogColor", ArgType.INT),
    35: SVDef("FogTableMode", ArgType.ENUM, ENUM_D3DFOGMODE),
    36: SVDef("FogStart", ArgType.FLOAT),
    37: SVDef("FogEnd", ArgType.FLOAT),
    38: SVDef("FogDensity", ArgType.FLOAT),
    48: SVDef("RangeFogEnable", ArgType.BOOL),
    52: SVDef("StencilEnable", ArgType.BOOL),
    53: SVDef("StencilFail", ArgType.ENUM, ENUM_D3DSTENCILOP),
    54: SVDef("StencilZFail", ArgType.ENUM, ENUM_D3DSTENCILOP),
    55: SVDef("StencilPass", ArgType.ENUM, ENUM_D3DSTENCILOP),
    56: SVDef("StencilFunc", ArgType.ENUM, ENUM_D3DCMP),
    57: SVDef("StencilRef", ArgType.INT),
    58: SVDef("StencilMask", ArgType.INT),
    59: SVDef("StencilWriteMask", ArgType.INT),
    60: SVDef("TextureFactor", ArgType.INT),
    136: SVDef("Clipping", ArgType.BOOL),
    137: SVDef("Lighting", ArgType.BOOL),
    139: SVDef("Ambient", ArgType.INT),
    140: SVDef("FogVertexMode", ArgType.ENUM, ENUM_D3DFOGMODE),
    141: SVDef("ColorVertex", ArgType.BOOL),
    142: SVDef("LocalViewer", ArgType.BOOL),
    143: SVDef("NormalizeNormals", ArgType.BOOL),
    145: SVDef("DiffuseMaterialSource", ArgType.ENUM, ENUM_D3DMATSOURCE),
    146: SVDef("SpecularMaterialSource", ArgType.ENUM, ENUM_D3DMATSOURCE),
    147: SVDef("AmbientMaterialSource", ArgType.ENUM, ENUM_D3DMATSOURCE),
    148: SVDef("EmissiveMaterialSource", ArgType.ENUM, ENUM_D3DMATSOURCE),
    151: SVDef("VertexBlend", ArgType.ENUM, ENUM_D3DVERTEXBLEND),
    152: SVDef("ClipPlaneEnable", ArgType.BITMASK, MASK_CLIPPLANES),
    154: SVDef("PointSize", ArgType.FLOAT),
    155: SVDef("PointSize_Min", ArgType.FLOAT),
    156: SVDef("PointSpriteEnable", ArgType.BOOL),
    157: SVDef("PointScaleEnable", ArgType.BOOL),
    158: SVDef("PointScale_A", ArgType.FLOAT),
    159: SVDef("PointScale_B", ArgType.FLOAT),
    160: SVDef("PointScale_C", ArgType.FLOAT),
    161: SVDef("MultisampleAntialias", ArgType.BOOL),
    162: SVDef("MultisampleMask", ArgType.INT),
    163: SVDef("PatchEdgeStyle", ArgType.ENUM, ENUM_D3DPATCHEDGE),
    165: SVDef("DebugMonitorToken", ArgType.ENUM, ENUM_D3DDMT),
    166: SVDef("PointSize_Max", ArgType.FLOAT),
    167: SVDef("IndexedVertexBlendEnable", ArgType.BOOL),
    168: SVDef("ColorWriteEnable", ArgType.BITMASK, MASK_COLOUR),
    170: SVDef("TweenFactor", ArgType.FLOAT),
    171: SVDef("BlendOp", ArgType.ENUM, ENUM_D3DBLENDOP),
    172: SVDef("PositionDegree", ArgType.ENUM, ENUM_D3DDEGREE),
    173: SVDef("NormalDegree", ArgType.ENUM, ENUM_D3DDEGREE),
    174: SVDef("ScissorTestEnable", ArgType.BOOL),
    175: SVDef("SlopeScaleDepthBias", ArgType.FLOAT),
    176: SVDef("AntialiasedLineEnable", ArgType.BOOL),
    178: SVDef("MinTessellationLevel", ArgType.FLOAT),
    179: SVDef("MaxTessellationLevel", ArgType.FLOAT),
    # Floats per the docs; vendor fourcc hacks (e.g. ATI R2VB in
    # AdaptiveTess_Y) still display as fourcc via describe_state_var.
    180: SVDef("AdaptiveTess_X", ArgType.FLOAT),
    181: SVDef("AdaptiveTess_Y", ArgType.FLOAT),
    182: SVDef("AdaptiveTess_Z", ArgType.FLOAT),
    183: SVDef("AdaptiveTess_W", ArgType.FLOAT),
    184: SVDef("EnableAdaptiveTessellation", ArgType.BOOL),
    185: SVDef("TwoSidedStencilMode", ArgType.BOOL),
    186: SVDef("CCW_StencilFail", ArgType.ENUM, ENUM_D3DSTENCILOP),
    187: SVDef("CCW_StencilZFail", ArgType.ENUM, ENUM_D3DSTENCILOP),
    188: SVDef("CCW_StencilPass", ArgType.ENUM, ENUM_D3DSTENCILOP),
    189: SVDef("CCW_StencilFunc", ArgType.ENUM, ENUM_D3DCMP),
    190: SVDef("ColorWriteEnable1", ArgType.BITMASK, MASK_COLOUR),
    191: SVDef("ColorWriteEnable2", ArgType.BITMASK, MASK_COLOUR),
    192: SVDef("ColorWriteEnable3", ArgType.BITMASK, MASK_COLOUR),
    193: SVDef("BlendFactor", ArgType.INT),
    194: SVDef("SRGBWriteEnable", ArgType.BOOL),
    195: SVDef("DepthBias", ArgType.FLOAT),
    206: SVDef("SeparateAlphaBlendEnable", ArgType.BOOL),
    207: SVDef("SrcBlendAlpha", ArgType.ENUM, ENUM_D3DBLEND),
    208: SVDef("DestBlendAlpha", ArgType.ENUM, ENUM_D3DBLEND),
    209: SVDef("BlendOpAlpha", ArgType.ENUM, ENUM_D3DBLENDOP),
}

# D3DRS_WRAP0-7 = 128-135, D3DRS_WRAP8-15 = 198-205
for _i in range(8):
    SV_TYPES_D3D[128 + _i] = SVDef("Wrap%d" % _i, ArgType.BITMASK, MASK_WRAPCOORD)
    SV_TYPES_D3D[198 + _i] = SVDef("Wrap%d" % (8 + _i), ArgType.BITMASK, MASK_WRAPCOORD)

SV_TYPES_TEX_D3D = {
    1: SVDef("AddressU", ArgType.ENUM, ENUM_D3DADDRESS),
    2: SVDef("AddressV", ArgType.ENUM, ENUM_D3DADDRESS),
    3: SVDef("AddressW", ArgType.ENUM, ENUM_D3DADDRESS),
    4: SVDef("BorderColor", ArgType.INT),
    5: SVDef("MagFilter", ArgType.ENUM, ENUM_D3DFILTER),
    6: SVDef("MinFilter", ArgType.ENUM, ENUM_D3DFILTER),
    7: SVDef("MipFilter", ArgType.ENUM, ENUM_D3DFILTER),
    8: SVDef("MipMapLODBias", ArgType.FLOAT),  # vendor fourcc hacks land here too
    9: SVDef("MaxMipLevel", ArgType.INT),
    10: SVDef("MaxAnisotropy", ArgType.INT),
    11: SVDef("SRGBTexture", ArgType.BOOL),
    12: SVDef("ElementIndex", ArgType.INT),
    13: SVDef("DMapOffset", ArgType.INT),
}


# D3D11-layout files: state var IDs are *not* raw D3D11 API values (D3D11 has
# no per-state setter like D3D9's SetRenderState - states are grouped into
# immutable rasterizer/depth-stencil/blend descriptor objects). They are
# instead a compact, engine-specific id space Diesel uses for its D3D11
# backend (observed range across deferred_lighting.d3d11 and base.d3d11: 1-46).
#
# The id space is the *field index* within each D3D11 descriptor struct, at a
# per-descriptor base:
#
#   base  0  D3D11_RASTERIZER_DESC    1=CullMode, 3=DepthBias,
#                                     4=DepthBiasClamp, 5=SlopeScaledDepthBias,
#                                     8=MultisampleEnable
#   base 10  D3D11_DEPTH_STENCIL_DESC 10..19 = DepthEnable, DepthWriteMask,
#                                     DepthFunc, StencilEnable,
#                                     StencilReadMask, StencilWriteMask,
#                                     FrontFace.{Fail,DepthFail,Pass,Func}Op;
#                                     20-23 would be BackFace (never set here),
#                                     and 24 is StencilRef - not a desc field
#                                     at all but the OMSetDepthStencilState
#                                     argument, appended after the 14 fields
#   base 28  D3D11_BLEND_DESC         28=AlphaToCoverageEnable, 29=BlendEnable,
#                                     37-39=SrcBlend/DestBlend/BlendOp,
#                                     44-46=RenderTargetWriteMask[0..2]
#
# That model is pinned at 15 independent points by diffing the d3d11 and d3d9
# builds of the same package: passes that exist in both under the same
# (render-template, mode, pass-index) key emit their state vars in the same
# canonical order, so aligning the two sequences from the *head* yields the
# mapping. base.d3d11.shaders vs base246.1.shaders shares 2771 such passes
# (deferred_lighting shares 30), and every id votes unanimously.
#
# Most D3D11 enums are numbered identically to their D3D9 counterparts
# (D3D11_COMPARISON_FUNC/_STENCIL_OP/_BLEND/_BLEND_OP/_COLOR_WRITE_ENABLE all
# match), which is why those values compare bit-identical across the two
# builds and can reuse the D3D9 SVDefs below. The exceptions get their own
# definitions in _SV_TYPES_D3D11_OWN.
#
# Correction, from base.d3d11 vs base246.1: ids 3 and 5 were previously read as
# SRGBWriteEnable and SeparateAlphaBlendEnable. Those came from aligning the
# unequal-length deferred_lighting passes from the *tail* instead of the head -
# the d3d9 side carries 7 extra states in the middle (DepthBias,
# SlopeScaleDepthBias, FogEnable, SpecularEnable, AdaptiveTess_X/Z/W) that the
# D3D11 backend has no use for, so tail-anchoring shifted the last three pairs.
# In deferred_lighting every state involved happened to be 0, so the two
# alignments were indistinguishable there. base.d3d11 breaks the tie outright:
# in its 1091 depth-biased shared passes id 3 = 255 and id 5 = 2.5f, while d3d9
# SRGBWriteEnable and SeparateAlphaBlendEnable are 0 in all 5207 of its passes
# and DepthBias/SlopeScaleDepthBias are 0.0002f/2.5f. Id 5 matches d3d9
# SlopeScaleDepthBias bit-for-bit in all 2771 shared passes; id 3 tracks
# DepthBias's zero/non-zero exactly but is an *int* (255) where D3D9 uses a
# float - the documented D3D9-to-D3D11 difference for this state.
#
# Id 40 was the third pair of that same refuted tail alignment (it read as
# AlphaTestEnable) so it has no evidential basis left. It is 0 in all 2776
# passes of base.d3d11 and all 30 of deferred_lighting.d3d11, and every d3d9
# state it could correspond to is likewise always 0, so this diff can neither
# confirm nor place it; the field-index model puts it at blend field 40, which
# would be RenderTarget[0].SrcBlendAlpha. Left to the raw-number fallback
# rather than displayed under a name that may be wrong.

ENUM_D3D11_CULL = [None, "None", "Front", "Back"]


def _d3d11_filter_names():
    """Sparse enum table for D3D11_FILTER, which is a bitfield rather than a
    sequence: bit0 = mip linear, bit2 = mag linear, bit4 = min linear,
    0x40 = anisotropic, 0x80 = comparison. (D3D11.1's 0x100/0x180 minimum and
    maximum reduction filters are not used by Diesel.)"""
    names = [None] * 0xD6
    for cmp_bit, cmp_name in ((0, ""), (0x80, "Comparison ")):
        for mn, mn_name in ((0, "Point"), (0x10, "Linear")):
            for mg, mg_name in ((0, "Point"), (0x04, "Linear")):
                for mp, mp_name in ((0, "Point"), (0x01, "Linear")):
                    names[cmp_bit | mn | mg | mp] = "%sMin %s, Mag %s, Mip %s" % (
                        cmp_name, mn_name, mg_name, mp_name)
        names[cmp_bit | 0x55] = cmp_name + "Anisotropic"
    return names


ENUM_D3D11_FILTER = _d3d11_filter_names()

# D3D11 ids that denote the same state as a D3D9 render state *and* share its
# value encoding, so the name/type/enum can be single-sourced from the D3D9
# table and the two stay in sync.
_D3D11_TO_D3D9 = {
    5: 175,   # SlopeScaleDepthBias
    11: 14,   # ZWriteEnable    (D3D11 DepthWriteMask ZERO/ALL = 0/1)
    12: 23,   # ZFunc
    13: 52,   # StencilEnable
    14: 58,   # StencilMask     (D3D11 StencilReadMask)
    15: 59,   # StencilWriteMask
    16: 53,   # StencilFail     (FrontFace.StencilFailOp)
    17: 54,   # StencilZFail    (FrontFace.StencilDepthFailOp)
    18: 55,   # StencilPass     (FrontFace.StencilPassOp)
    19: 56,   # StencilFunc     (FrontFace.StencilFunc)
    24: 57,   # StencilRef
    29: 27,   # AlphaBlendEnable(RenderTarget[0].BlendEnable)
    37: 19,   # SrcBlend
    38: 20,   # DestBlend
    39: 171,  # BlendOp
    44: 168,  # ColorWriteEnable (RenderTargetWriteMask[0])
    45: 190,  # ColorWriteEnable1
    46: 191,  # ColorWriteEnable2
}

# D3D11 ids needing their own definition: a differently-numbered enum, a
# different value type, or no D3D9 counterpart at all.
_SV_TYPES_D3D11_OWN = {
    # D3D11_CULL_MODE is None/Front/Back where D3DCULL is None/CW/CCW, so the
    # same id needs a different enum. Confirmed 1:1 against d3d9 CullMode over
    # the 2771 shared passes: 1 <-> 1 (None) x332, 3 (Back) <-> 2 (CW) x2439.
    1: SVDef("CullMode", ArgType.ENUM, ENUM_D3D11_CULL),
    # Int in D3D11 (a raw depth-unit offset), float in D3D9.
    3: SVDef("DepthBias", ArgType.INT),
    # D3D11-only: D3D9 has no depth-bias clamp. Emitted only in the 1093
    # depth-biased passes of base.d3d11, always 0.1f, and at exactly the
    # rasterizer-desc index the model predicts for it.
    4: SVDef("DepthBiasClamp", ArgType.FLOAT),
    # Plain BOOL in D3D11; the D3D9 id is the tri-state D3DZBUFFERTYPE.
    10: SVDef("ZEnable", ArgType.BOOL),
    # 8 and 28 are always 0 in both sample files, so the diff cannot
    # discriminate them - they are named from the field-index model alone
    # (rasterizer field 8, blend field 0) and both are BOOLs there.
    8: SVDef("MultisampleEnable", ArgType.BOOL),
    28: SVDef("AlphaToCoverageEnable", ArgType.BOOL),
}

SV_TYPES_D3D11 = {d11: SV_TYPES_D3D[d9] for d11, d9 in _D3D11_TO_D3D9.items()}
SV_TYPES_D3D11.update(_SV_TYPES_D3D11_OWN)

# Texture/sampler vars: D3D11 sampler descriptors are shaped differently from
# D3D9's per-state model. id1/id2 carry the same D3DTADDRESS enum values as
# D3D9 AddressU/AddressV. id0 is a *packed* D3D11_FILTER that collapses D3D9's
# separate Mag/Min/MipFilter; aligning the 3729 texture blocks that pair up
# across base.d3d11 and base246.1 maps it unanimously (0x55 <-> all three D3D9
# filters Anisotropic, 0x15 <-> all three Linear, 0x00 <-> all three Point),
# and deferred_lighting.d3d11 additionally uses 0x94 (comparison, min/mag
# linear, mip point) for its shadow samplers. id13 is SRGBTexture: it is
# present with value 1 in exactly the 2308 of those blocks whose D3D9 side has
# SRGBTexture=1 and omitted in exactly the 1421 where it is 0. id6 appears in
# neither sample file and is left to the raw-number fallback.
SV_TYPES_TEX_D3D11 = {
    0: SVDef("Filter", ArgType.ENUM, ENUM_D3D11_FILTER),
    1: SV_TYPES_TEX_D3D[1],   # AddressU
    2: SV_TYPES_TEX_D3D[2],   # AddressV
    13: SV_TYPES_TEX_D3D[11],  # SRGBTexture
}


def tables_for_pass(shader_pass):
    """(state var table, texture var table) appropriate for a pass's layout."""
    if getattr(shader_pass, "layout", "d3d9") == "d3d11":
        return SV_TYPES_D3D11, SV_TYPES_TEX_D3D11
    return SV_TYPES_D3D, SV_TYPES_TEX_D3D


def fourcc_or_none(value):
    """Render a u32 as a fourcc string if all four bytes are printable ASCII."""
    b = struct.pack("<I", value)
    if all(32 <= c < 127 for c in b):
        return b.decode("ascii")
    return None


def describe_state_var(sv, table):
    """(name, value-string) for display."""
    sdef = table.get(sv.id)
    name = sdef.name if sdef else "id %d" % sv.id
    if sv.flag != 0:
        return name, "idstring %016x" % sv.val8
    fcc = fourcc_or_none(sv.val4)
    if sdef:
        # Vendor fourcc hacks (e.g. 'GET1' in MipMapLODBias) hide in
        # float/int-typed vars; show those as fourcc rather than garbage.
        if fcc and sdef.arg_type in (ArgType.FLOAT, ArgType.INT):
            return name, "'%s' (0x%08x)" % (fcc, sv.val4)
        decoded = sdef.decode(sv)
        if decoded is not None and not decoded.startswith("<invalid"):
            return name, decoded
    if fcc:
        return name, "'%s' (0x%08x)" % (fcc, sv.val4)
    return name, "0x%08x (%d)" % (sv.val4, sv.val4)


def parse_var_value(sv, sdef, text, hash_fn):
    """Parse user text into a raw value for a StateVar (u32, or u64 for
    idstrings). hash_fn turns a string into a Diesel hash. Raises ValueError."""
    text = text.strip()
    if sv.flag != 0:
        if len(text) == 16 and all(c in "0123456789abcdefABCDEF" for c in text):
            return int(text, 16)
        return hash_fn(text)
    if sdef and sdef.arg_type in (ArgType.ENUM, ArgType.BOOL, ArgType.BITMASK,
                                  ArgType.FLOAT):
        try:
            return sdef.encode(text)
        except ValueError:
            if sdef.arg_type != ArgType.FLOAT:
                raise
    # Free-form integer / fourcc
    if len(text) == 4 and not text.startswith("0x") and \
            not text.lstrip("-").isdigit() and all(32 <= ord(c) < 127 for c in text):
        return struct.unpack("<I", text.encode("ascii"))[0]
    try:
        return int(text, 0) & 0xFFFFFFFF
    except ValueError:
        raise ValueError("Enter a number (e.g. 3 or 0x1f) or a 4-char fourcc")
