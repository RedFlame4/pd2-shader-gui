"""Read-only disassembler for D3D11 shader bytecode (DXBC container, shader
model 4/5 tokenized program format used by SHEX/SHDR chunks).

Produces fxc-style assembly listings, plus an RDEF-derived header (constant
buffer layout, bound textures/samplers/cbuffers) analogous to the D3D9
disassembler's CTAB header. Unknown opcodes degrade to raw hex lines; any
exception is caught by disassemble() so callers always get text - same
contract as d3d9_disasm.disassemble().
"""

import struct

# D3D10_SB_OPCODE_TYPE -> mnemonic. Verified against every VS/PS blob in
# deferred_lighting.d3d11.shaders: DCL_RESOURCE/DCL_SAMPLER/DCL_CONSTANT_BUFFER
# counts and dcl_resource's decoded dimension/name were cross-checked against
# each shader's own RDEF resource-binding table. Unlisted/exotic opcodes
# (compute/hull/domain/geometry-only, atomics, doubles) fall back to a raw
# hex line rather than risk a wrong mnemonic - this file only ever contains
# VS/PS bytecode.
OPCODES = {
    0x00: "add", 0x01: "and", 0x02: "break", 0x03: "breakc", 0x04: "call",
    0x05: "callc", 0x06: "case", 0x07: "continue", 0x08: "continuec",
    0x09: "cut", 0x0A: "default", 0x0B: "deriv_rtx", 0x0C: "deriv_rty",
    0x0D: "discard", 0x0E: "div", 0x0F: "dp2", 0x10: "dp3", 0x11: "dp4",
    0x12: "else", 0x13: "emit", 0x14: "emitThenCut", 0x15: "endif",
    0x16: "endloop", 0x17: "endswitch", 0x18: "eq", 0x19: "exp", 0x1A: "frc",
    0x1B: "ftoi", 0x1C: "ftou", 0x1D: "ge", 0x1E: "iadd", 0x1F: "if",
    0x20: "ieq", 0x21: "ige", 0x22: "ilt", 0x23: "imad", 0x24: "imax",
    0x25: "imin", 0x26: "imul", 0x27: "ine", 0x28: "ineg", 0x29: "ishl",
    0x2A: "ishr", 0x2B: "itof", 0x2C: "label", 0x2D: "ld", 0x2E: "ld_ms",
    0x2F: "log", 0x30: "loop", 0x31: "lt", 0x32: "mad", 0x33: "min",
    0x34: "max", 0x36: "mov", 0x37: "movc", 0x38: "mul", 0x39: "ne",
    0x3A: "nop", 0x3B: "not", 0x3C: "or", 0x3D: "resinfo", 0x3E: "ret",
    0x3F: "retc", 0x40: "round_ne", 0x41: "round_ni", 0x42: "round_pi",
    0x43: "round_z", 0x44: "rsq", 0x45: "sample", 0x46: "sample_c",
    0x47: "sample_c_lz", 0x48: "sample_l", 0x49: "sample_d", 0x4A: "sample_b",
    0x4B: "sqrt", 0x4C: "switch", 0x4D: "sincos", 0x4E: "udiv", 0x4F: "ult",
    0x50: "uge", 0x51: "umul", 0x52: "umad", 0x53: "umax", 0x54: "umin",
    0x55: "ushr", 0x56: "utof", 0x57: "xor",
    0x6B: "lod", 0x6C: "gather4", 0x6D: "samplePos", 0x6E: "sampleInfo",
    0x91: "bufinfo",
    # 0x8c is used by this file (bitfield-ish 1-dest/4-src shape) but its
    # exact identity/operand order isn't confidently pinned down - left
    # unmapped so it degrades to a raw hex line instead of a guessed mnemonic.
}

# Opcodes with their own custom rendering (declarations, control-flow test
# suffixes, saturate flag, and the icb literal-data block).
DCL_TEMPS = 0x68
DCL_INDEXABLE_TEMP = 0x69
DCL_GLOBAL_FLAGS = 0x6A
DCL_RESOURCE = 0x58
DCL_CONSTANT_BUFFER = 0x59
DCL_SAMPLER = 0x5A
DCL_INPUT = 0x5F
DCL_INPUT_SGV = 0x60
DCL_INPUT_SIV = 0x61
DCL_INPUT_PS = 0x62
DCL_INPUT_PS_SGV = 0x63
DCL_INPUT_PS_SIV = 0x64
DCL_OUTPUT = 0x65
DCL_OUTPUT_SGV = 0x66
DCL_OUTPUT_SIV = 0x67
CUSTOMDATA = 0x35
CUSTOMDATA_ICB = 3  # D3D10_SB_CUSTOMDATA_CLASS: immediate constant buffer

DCL_OPCODES = {DCL_RESOURCE, DCL_CONSTANT_BUFFER, DCL_SAMPLER, DCL_INPUT,
               DCL_INPUT_SGV, DCL_INPUT_SIV, DCL_INPUT_PS, DCL_INPUT_PS_SGV,
               DCL_INPUT_PS_SIV, DCL_OUTPUT, DCL_OUTPUT_SGV, DCL_OUTPUT_SIV,
               DCL_TEMPS, DCL_INDEXABLE_TEMP, DCL_GLOBAL_FLAGS}

# Test-condition suffix (_z/_nz), bit 18 of the opcode token.
TEST_OPCODES = {0x03, 0x05, 0x08, 0x0D, 0x1F, 0x3F}  # breakc,callc,continuec,discard,if,retc

# Mnemonics whose operands (including literal l(...) immediates) are
# integer/bitwise, not float - an IMMEDIATE32 token is just an untyped
# 32-bit word, so which way to print it depends on the consuming
# instruction, not the encoding itself.
INT_TYPED_MNEMONICS = {
    "and", "or", "not", "xor", "ieq", "ige", "ilt", "ine", "ineg", "imad",
    "imax", "imin", "imul", "ishl", "ishr", "itof", "udiv", "ult", "uge",
    "umul", "umad", "umax", "umin", "ushr", "iadd",
}

# D3D10_SB_RESOURCE_DIMENSION, packed directly into bits 11-15 of
# DCL_RESOURCE's own opcode token (verified against this file's bytecode -
# a full-screen depth/diffuse sample pass decodes to dim=3 "texture2d",
# matching what the pass actually binds).
RESOURCE_DIMS = {0: "unknown", 1: "buffer", 2: "texture1d", 3: "texture2d",
                 4: "texture3d", 5: "texturecube", 6: "texture1darray",
                 7: "texture2darray", 8: "texture2dms", 9: "texture2dmsarray",
                 10: "texturecubearray", 11: "raw_buffer", 12: "structured_buffer"}
RETURN_TYPES = {1: "unorm", 2: "snorm", 3: "sint", 4: "uint", 5: "float",
                6: "mixed", 7: "double", 8: "continued", 9: "unused"}

REGTYPE_PREFIX = {0: "r", 1: "v", 2: "o", 3: "x", 6: "s", 7: "t", 8: "cb",
                  10: "l", 11: "vPrim", 12: "oDepth", 13: "null"}
MODIFIER_FMT = {1: "-{}", 2: "abs({})", 3: "-abs({})"}
SWIZZLE = "xyzw"


class _BoundsError(Exception):
    pass


def _decode_operand(words, p, end, as_int=False):
    if p >= end:
        raise _BoundsError("operand past instruction end")
    tok0 = words[p]
    p += 1
    num_comp = tok0 & 0x3
    sel_mode = (tok0 >> 2) & 0x3
    comp_bits = (tok0 >> 4) & 0xFF
    regtype = (tok0 >> 12) & 0xFF
    index_dim = (tok0 >> 20) & 0x3
    idx_reprs = [(tok0 >> 22) & 0x7, (tok0 >> 25) & 0x7, (tok0 >> 28) & 0x7]
    extended = (tok0 >> 31) & 1

    modfmt = "{}"
    while extended:
        ext_tok = words[p]
        p += 1
        extended = (ext_tok >> 31) & 1
        if ext_tok & 0x3F == 1:  # MODIFIER
            modfmt = MODIFIER_FMT.get((ext_tok >> 6) & 0xFF, "{}")

    indices = []
    for d in range(index_dim):
        rep = idx_reprs[d]
        imm = rel = None
        if rep in (0, 3):
            imm = words[p]; p += 1
        elif rep in (1, 4):
            imm = words[p] | (words[p + 1] << 32); p += 2
        if rep in (2, 3, 4):
            rel, p = _decode_operand(words, p, end)
        indices.append((imm, rel))

    if regtype in (4, 5):  # IMMEDIATE32 / IMMEDIATE64
        n = 4 if num_comp == 2 else 1
        if regtype == 4:
            raw = words[p:p + n]
            p += n
            # Immediates are untyped 32-bit words; whether they mean float
            # or int depends on the consuming instruction (as_int, set by
            # the caller from the opcode's mnemonic), not the encoding.
            vals = raw if as_int else struct.unpack_from(
                "<%df" % n, struct.pack("<%dI" % n, *raw))
        else:
            raw = words[p:p + 2 * n]
            p += 2 * n
            vals = struct.unpack("<%dd" % n, struct.pack("<%dI" % (2 * n), *raw))
        text = "l(" + ",".join(("%d" if as_int else "%g") % v for v in vals) + ")"
    else:
        name = REGTYPE_PREFIX.get(regtype, "reg%d" % regtype)
        if not indices:
            text = name
        else:
            imm0, rel0 = indices[0]
            text = name + (str(imm0) if rel0 is None else "[%s]" % rel0)
            for imm, rel in indices[1:]:
                text += ("[%s%s]" % (rel, "+%d" % imm if imm else "")
                         if rel is not None else "[%d]" % imm)

    if num_comp == 2:
        if sel_mode == 0:  # write mask (destinations)
            comps = "".join(c for i, c in enumerate(SWIZZLE) if comp_bits & (1 << i))
            if comps and comps != SWIZZLE:
                text += "." + comps
        elif sel_mode == 1:  # swizzle (sources)
            comps = "".join(SWIZZLE[(comp_bits >> (2 * i)) & 3] for i in range(4))
            if comps != SWIZZLE:
                text += "." + comps
        elif sel_mode == 2:  # select-1 component
            text += "." + SWIZZLE[comp_bits & 3]
    return modfmt.format(text), p


def _skip_extended_opcode_tokens(words, p):
    """Opcode-level extended tokens (chained via each token's own bit 31) -
    e.g. SAMPLE_CONTROLS immediate texel offsets on sample/sample_l/gather4.
    Their payloads aren't rendered; just consume them so operand decoding
    resumes at the right word."""
    tok = words[p - 1]
    while (tok >> 31) & 1:
        tok = words[p]
        p += 1
    return p


def _dcl_line(words, opcode_pos, p, end, opcode):
    """Render a declaration instruction. Returns a listing line."""
    cur = p
    if opcode == DCL_TEMPS:
        return "dcl_temps %d" % words[cur]
    if opcode == DCL_INDEXABLE_TEMP:
        reg, size, comps = words[cur], words[cur + 1], words[cur + 2]
        return "dcl_indexableTemp x%d[%d], %d" % (reg, size, comps)
    if opcode == DCL_GLOBAL_FLAGS:
        return "dcl_globalFlags 0x%x" % ((words[opcode_pos] >> 11) & 0xFF)
    if opcode == DCL_RESOURCE:
        # RESOURCE_DIMENSION is packed directly into the DCL_RESOURCE opcode
        # token itself (bits 11-15), not behind an extended-opcode token.
        dim_bits = (words[opcode_pos] >> 11) & 0x1F
        operand, cur = _decode_operand(words, cur, end)
        dim = RESOURCE_DIMS.get(dim_bits, "dim%s" % dim_bits)
        types = "unknown"
        if cur < end:
            rt = words[cur]
            types = ",".join(RETURN_TYPES.get((rt >> (4 * i)) & 0xF, "?") for i in range(4))
        return "dcl_resource_%s (%s) %s" % (dim, types, operand)
    if opcode == DCL_CONSTANT_BUFFER:
        operand, cur = _decode_operand(words, cur, end)
        return "dcl_constantbuffer %s" % operand
    if opcode == DCL_SAMPLER:
        operand, cur = _decode_operand(words, cur, end)
        return "dcl_sampler %s" % operand
    # Input/output declarations: just show the register (+mask); usage
    # semantics (SGV/SIV system-value names, interpolation mode) are left
    # out rather than risk a wrong label.
    operand, cur = _decode_operand(words, cur, end)
    name = {DCL_INPUT: "dcl_input", DCL_INPUT_SGV: "dcl_input_sgv",
            DCL_INPUT_SIV: "dcl_input_siv", DCL_INPUT_PS: "dcl_input_ps",
            DCL_INPUT_PS_SGV: "dcl_input_ps_sgv", DCL_INPUT_PS_SIV: "dcl_input_ps_siv",
            DCL_OUTPUT: "dcl_output", DCL_OUTPUT_SGV: "dcl_output_sgv",
            DCL_OUTPUT_SIV: "dcl_output_siv"}.get(opcode, "dcl_%#x" % opcode)
    return "%s %s" % (name, operand)


def _customdata_line(words, p):
    """CUSTOMDATA (icb) has its own encoding: opcode token, then a length
    dword, then raw payload - not the usual ilen-in-opcode-token scheme."""
    length = words[p]
    payload = words[p + 1:p + length]
    floats = struct.unpack("<%df" % len(payload), struct.pack("<%dI" % len(payload), *payload))
    lines = ["icb = {"]
    for i in range(0, len(floats), 4):
        lines.append("  {" + ", ".join("%g" % v for v in floats[i:i + 4]) + "},")
    lines.append("}")
    return lines, p + length


def _instruction_line(words, p, opcode, ilen):
    end = p + ilen
    cur = _skip_extended_opcode_tokens(words, p + 1)

    if opcode in DCL_OPCODES:
        return _dcl_line(words, p, cur, end, opcode)

    mnemonic = OPCODES.get(opcode)
    if mnemonic is None:
        raise ValueError("unknown opcode %#x" % opcode)

    opctl = words[p]
    if opcode in TEST_OPCODES:
        mnemonic += "_nz" if (opctl >> 18) & 1 else "_z"
    elif (opctl >> 13) & 1:  # saturate
        mnemonic += "_sat"

    as_int = mnemonic in INT_TYPED_MNEMONICS
    ops = []
    while cur < end:
        text, cur = _decode_operand(words, cur, end, as_int=as_int)
        ops.append(text)
    return mnemonic + ((" " + ", ".join(ops)) if ops else "")


INDENT_OPEN = {"if_z", "if_nz", "loop", "switch"}
INDENT_CLOSE = {"endif", "endloop", "endswitch"}


def _disassemble_tokens(words, start, length_dwords):
    lines = []
    indent = 0
    p = start
    end = start + length_dwords
    while p < end:
        tok = words[p]
        opcode = tok & 0x7FF
        if opcode == CUSTOMDATA:
            cls = (tok >> 11) & 0x1F
            if cls == CUSTOMDATA_ICB:
                block, p = _customdata_line(words, p + 1)
                lines.extend(block)
                continue
            length = words[p + 1]
            lines.append("// customdata (class %d, %d dwords)" % (cls, length))
            p += length
            continue

        ilen = (tok >> 24) & 0x7F
        if ilen == 0:
            break
        try:
            line = _instruction_line(words, p, opcode, ilen)
            base = line.split()[0].split("_sat")[0]
            if base in INDENT_CLOSE:
                indent = max(0, indent - 1)
            lines.append("  " * indent + line)
            if base in INDENT_OPEN or base in ("else",):
                indent += 1
        except Exception as e:  # noqa: BLE001 - degrade this instruction only
            lines.append("  // opcode %#x (%d dwords): %s" % (opcode, ilen, e))
        p += ilen
    return lines


# --- DXBC container parsing ---

def _chunks(blob):
    n_chunks = struct.unpack_from("<I", blob, 28)[0]
    offsets = struct.unpack_from("<%dI" % n_chunks, blob, 32)
    for off in offsets:
        tag = blob[off:off + 4]
        size = struct.unpack_from("<I", blob, off + 4)[0]
        yield tag, off + 8, size


def _find_chunk(blob, tag):
    for t, off, size in _chunks(blob):
        if t == tag:
            return off, size
    return None, None


def _cstr(buf, off):
    end = buf.index(b"\0", off)
    return buf[off:end].decode("latin-1")


def _parse_rdef(blob):
    """-> dict with creator/target/cbuffers/resources, or None."""
    off, size = _find_chunk(blob, b"RDEF")
    if off is None:
        return None
    rdef = blob[off:off + size]
    n_cb, cb_off, n_rb, rb_off, minor, major, prog_type, flags, creator_off = \
        struct.unpack_from("<IIIIBBHII", rdef, 0)
    is_rd11 = rdef[28:32] == b"RD11"
    var_stride = 40 if is_rd11 else 24

    cbuffers = []
    for i in range(n_cb):
        name_off, var_count, var_off, cb_size, cb_flags, cb_type = \
            struct.unpack_from("<6I", rdef, cb_off + i * 24)
        variables = []
        for v in range(var_count):
            vname_off, start, vsize, vflags = struct.unpack_from(
                "<4I", rdef, var_off + v * var_stride)
            variables.append((_cstr(rdef, vname_off), start, vsize))
        cbuffers.append((_cstr(rdef, name_off), cb_size, variables))

    resources = []
    for i in range(n_rb):
        name_off, rtype, ret_type, dim, num_samples, bind_point, bind_count, rflags = \
            struct.unpack_from("<8I", rdef, rb_off + i * 32)
        resources.append({"name": _cstr(rdef, name_off), "type": rtype,
                           "bind_point": bind_point})

    kind = {0xFFFF: "ps", 0xFFFE: "vs"}.get(prog_type, "shader")
    return {"creator": _cstr(rdef, creator_off), "profile": "%s_%d_%d" % (kind, major, minor),
            "cbuffers": cbuffers, "resources": resources}


# D3D_SHADER_INPUT_TYPE values we care about for labelling.
RTYPE_SAMPLER = 3
RTYPE_TEXTURE = 2


def resource_names(blob):
    """{bind_point: name} for texture (t#) resources, {} if unavailable."""
    try:
        info = _parse_rdef(blob)
        if info is None:
            return {}
        return {r["bind_point"]: r["name"] for r in info["resources"]
                if r["type"] == RTYPE_TEXTURE}
    except Exception:  # noqa: BLE001 - best-effort metadata only
        return {}


def sampler_names(blob):
    """{bind_point: name} for sampler (s#) resources, {} if unavailable.
    Mirrors d3d9_disasm.sampler_names' contract for shader_web.py."""
    try:
        info = _parse_rdef(blob)
        if info is None:
            return {}
        return {r["bind_point"]: r["name"] for r in info["resources"]
                if r["type"] == RTYPE_SAMPLER}
    except Exception:  # noqa: BLE001
        return {}


def _rdef_header(info):
    lines = ["// %s" % info["profile"], "// creator: %s" % info["creator"], "//"]
    for name, size, variables in info["cbuffers"]:
        lines.append("// cbuffer %s (%d bytes)" % (name, size))
        for vname, start, vsize in sorted(variables, key=lambda t: t[1]):
            lines.append("//   +%-5d %-5d %s" % (start, vsize, vname))
    if info["resources"]:
        lines.append("//")
        lines.append("// bound resources:")
        for r in sorted(info["resources"], key=lambda r: (r["type"], r["bind_point"])):
            kind = {RTYPE_TEXTURE: "texture", RTYPE_SAMPLER: "sampler"}.get(r["type"], "cbuffer")
            reg = {RTYPE_TEXTURE: "t", RTYPE_SAMPLER: "s"}.get(r["type"], "cb")
            lines.append("//   %s%-3d %-8s %s" % (reg, r["bind_point"], kind, r["name"]))
    return lines


def disassemble(blob):
    try:
        return _disassemble(blob)
    except Exception as e:  # noqa: BLE001 - must never block the UI
        return "// Disassembly failed: %s\n\n%s" % (e, _hexdump(blob))


def _disassemble(blob):
    if len(blob) < 4:
        return "// (empty shader)"
    if blob[:4] != b"DXBC":
        return "// Not DXBC (D3D10/11) bytecode (first bytes %r)\n\n%s" % (
            blob[:4], _hexdump(blob))

    header = []
    info = _parse_rdef(blob)
    if info:
        header = _rdef_header(info)

    off, size = _find_chunk(blob, b"SHEX")
    if off is None:
        off, size = _find_chunk(blob, b"SHDR")
    if off is None:
        out = list(header)
        out.append("// (no SHEX/SHDR chunk found)")
        return "\n".join(out)

    shex = blob[off:off + size]
    n_words = len(shex) // 4
    words = struct.unpack("<%dI" % n_words, shex[:n_words * 4])
    ver_tok, length_tok = words[0], words[1]
    minor, major = ver_tok & 0xF, (ver_tok >> 4) & 0xF
    prog_type = (ver_tok >> 16) & 0xFFFF
    kind = {0: "ps", 1: "vs", 2: "gs", 3: "hs", 4: "ds", 5: "cs"}.get(prog_type, "shader")

    lines = ["%s_%d_%d" % (kind, major, minor)]
    lines.extend(_disassemble_tokens(words, 2, min(length_tok, n_words) - 2))

    out = []
    if header:
        out.extend(header)
        out.append("")
    out.extend(lines)
    return "\n".join(out)


def _hexdump(blob, limit=4096):
    lines = []
    for i in range(0, min(len(blob), limit), 16):
        chunk = blob[i:i + 16]
        hexs = " ".join("%02x" % b for b in chunk)
        asc = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append("%08x  %-48s  %s" % (i, hexs, asc))
    if len(blob) > limit:
        lines.append("... (%d more bytes)" % (len(blob) - limit))
    return "\n".join(lines)
