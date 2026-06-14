"""Read-only disassembler for D3D9 shader bytecode (shader models 1-3).

Produces fxc-style assembly listings. Unknown opcodes degrade to raw hex
lines; any exception is caught by disassemble() so callers always get text.
"""

import struct

# D3DSIO opcode -> (mnemonic, dest count, source count)
OPCODES = {
    0x00: ("nop", 0, 0),
    0x01: ("mov", 1, 1),
    0x02: ("add", 1, 2),
    0x03: ("sub", 1, 2),
    0x04: ("mad", 1, 3),
    0x05: ("mul", 1, 2),
    0x06: ("rcp", 1, 1),
    0x07: ("rsq", 1, 1),
    0x08: ("dp3", 1, 2),
    0x09: ("dp4", 1, 2),
    0x0A: ("min", 1, 2),
    0x0B: ("max", 1, 2),
    0x0C: ("slt", 1, 2),
    0x0D: ("sge", 1, 2),
    0x0E: ("exp", 1, 1),
    0x0F: ("log", 1, 1),
    0x10: ("lit", 1, 1),
    0x11: ("dst", 1, 2),
    0x12: ("lrp", 1, 3),
    0x13: ("frc", 1, 1),
    0x14: ("m4x4", 1, 2),
    0x15: ("m4x3", 1, 2),
    0x16: ("m3x4", 1, 2),
    0x17: ("m3x3", 1, 2),
    0x18: ("m3x2", 1, 2),
    0x19: ("call", 0, 1),
    0x1A: ("callnz", 0, 2),
    0x1B: ("loop", 0, 2),
    0x1C: ("ret", 0, 0),
    0x1D: ("endloop", 0, 0),
    0x1E: ("label", 0, 1),
    0x1F: ("dcl", 1, 0),  # special handling
    0x20: ("pow", 1, 2),
    0x21: ("crs", 1, 2),
    0x22: ("sgn", 1, 3),
    0x23: ("abs", 1, 1),
    0x24: ("nrm", 1, 1),
    0x25: ("sincos", 1, 1),
    0x26: ("rep", 0, 1),
    0x27: ("endrep", 0, 0),
    0x28: ("if", 0, 1),
    0x29: ("if", 0, 2),  # ifc - comparison in control bits
    0x2A: ("else", 0, 0),
    0x2B: ("endif", 0, 0),
    0x2C: ("break", 0, 0),
    0x2D: ("break", 0, 2),  # breakc
    0x2E: ("mova", 1, 1),
    0x2F: ("defb", 1, 0),  # special
    0x30: ("defi", 1, 0),  # special
    0x40: ("texcoord", 1, 0),
    0x41: ("texkill", 1, 0),
    0x42: ("texld", 1, 2),
    0x43: ("texbem", 1, 1),
    0x44: ("texbeml", 1, 1),
    0x45: ("texreg2ar", 1, 1),
    0x46: ("texreg2gb", 1, 1),
    0x47: ("texm3x2pad", 1, 1),
    0x48: ("texm3x2tex", 1, 1),
    0x49: ("texm3x3pad", 1, 1),
    0x4A: ("texm3x3tex", 1, 1),
    0x4C: ("texm3x3spec", 1, 2),
    0x4D: ("texm3x3vspec", 1, 1),
    0x4E: ("expp", 1, 1),
    0x4F: ("logp", 1, 1),
    0x50: ("cnd", 1, 3),
    0x51: ("def", 1, 0),  # special
    0x52: ("texreg2rgb", 1, 1),
    0x53: ("texdp3tex", 1, 1),
    0x54: ("texm3x2depth", 1, 1),
    0x55: ("texdp3", 1, 1),
    0x56: ("texm3x3", 1, 1),
    0x57: ("texdepth", 1, 0),
    0x58: ("cmp", 1, 3),
    0x59: ("bem", 1, 2),
    0x5A: ("dp2add", 1, 3),
    0x5B: ("dsx", 1, 1),
    0x5C: ("dsy", 1, 1),
    0x5D: ("texldd", 1, 4),
    0x5E: ("setp", 1, 2),
    0x5F: ("texldl", 1, 2),
    0x60: ("breakp", 0, 1),
}

COMPARISONS = {1: "_gt", 2: "_eq", 3: "_ge", 4: "_lt", 5: "_ne", 6: "_le"}

USAGE_NAMES = ["position", "blendweight", "blendindices", "normal", "psize",
               "texcoord", "tangent", "binormal", "tessfactor", "positiont",
               "color", "fog", "depth", "sample"]

SAMPLER_TYPES = {1: "_1d", 2: "_2d", 3: "_cube", 4: "_volume"}

# Register-type numbers from the two split bitfields in param tokens
RT_TEMP, RT_INPUT, RT_CONST, RT_ADDR, RT_RASTOUT, RT_ATTROUT, RT_TEXCRDOUT, \
    RT_CONSTINT, RT_COLOROUT, RT_DEPTHOUT, RT_SAMPLER, RT_CONST2, RT_CONST3, \
    RT_CONST4, RT_CONSTBOOL, RT_LOOP, RT_TEMPFLOAT16, RT_MISCTYPE, RT_LABEL, \
    RT_PREDICATE = range(20)

RASTOUT_NAMES = {0: "oPos", 1: "oFog", 2: "oPts"}
MISC_NAMES = {0: "vPos", 1: "vFace"}

SRC_MODS = {
    0: "{}", 1: "-{}", 2: "{}_bias", 3: "-{}_bias", 4: "{}_bx2", 5: "-{}_bx2",
    6: "1-{}", 7: "{}_x2", 8: "-{}_x2", 9: "{}_dz", 10: "{}_dw",
    11: "{}_abs", 12: "-{}_abs", 13: "!{}",
}


def _reg_type(tok):
    return ((tok >> 28) & 0x7) | (((tok >> 11) & 0x3) << 3)


def _reg_name(rtype, num, is_ps, version):
    if rtype == RT_TEMP:
        return "r%d" % num
    if rtype == RT_INPUT:
        return "v%d" % num
    if rtype == RT_CONST:
        return "c%d" % num
    if rtype == RT_ADDR:  # a# in VS, t# in PS
        return ("t%d" % num) if is_ps else ("a%d" % num)
    if rtype == RT_RASTOUT:
        return RASTOUT_NAMES.get(num, "oRast%d" % num)
    if rtype == RT_ATTROUT:
        return "oD%d" % num
    if rtype == RT_TEXCRDOUT:  # o# in vs_3_0, oT# before
        return ("o%d" % num) if version >= (3, 0) else ("oT%d" % num)
    if rtype == RT_CONSTINT:
        return "i%d" % num
    if rtype == RT_COLOROUT:
        return "oC%d" % num
    if rtype == RT_DEPTHOUT:
        return "oDepth"
    if rtype == RT_SAMPLER:
        return "s%d" % num
    if rtype in (RT_CONST2, RT_CONST3, RT_CONST4):
        return "c%d" % (num + 2048 * (rtype - RT_CONST2 + 1))
    if rtype == RT_CONSTBOOL:
        return "b%d" % num
    if rtype == RT_LOOP:
        return "aL"
    if rtype == RT_MISCTYPE:
        return MISC_NAMES.get(num, "vMisc%d" % num)
    if rtype == RT_LABEL:
        return "l%d" % num
    if rtype == RT_PREDICATE:
        return "p%d" % num
    return "<reg%d:%d>" % (rtype, num)


def _swizzle(tok):
    comps = "xyzw"
    s = "".join(comps[(tok >> (16 + 2 * i)) & 3] for i in range(4))
    if s == "xyzw":
        return ""
    if s == s[0] * 4:
        return "." + s[0]
    return "." + s


def _write_mask(tok):
    mask = (tok >> 16) & 0xF
    if mask == 0xF:
        return ""
    return "." + "".join(c for i, c in enumerate("xyzw") if mask & (1 << i))


class _Stream:
    def __init__(self, blob):
        self.words = struct.unpack("<%dI" % (len(blob) // 4), blob[:len(blob) // 4 * 4])
        self.pos = 0

    def peek(self):
        return self.words[self.pos]

    def next(self):
        w = self.words[self.pos]
        self.pos += 1
        return w

    def eof(self):
        return self.pos >= len(self.words)


def _fmt_dest(tok, is_ps, version):
    rtype = _reg_type(tok)
    name = _reg_name(rtype, tok & 0x7FF, is_ps, version)
    mods = (tok >> 20) & 0xF
    suffix = ""
    if mods & 1:
        suffix += "_sat"
    if mods & 2:
        suffix += "_pp"
    if mods & 4:
        suffix += "_centroid"
    return name, suffix, _write_mask(tok)


def _fmt_source(tok, rel_tok, is_ps, version):
    rtype = _reg_type(tok)
    name = _reg_name(rtype, tok & 0x7FF, is_ps, version)
    if rel_tok is not None:
        idx = _reg_name(_reg_type(rel_tok), rel_tok & 0x7FF, is_ps, version)
        idx += _swizzle(rel_tok) or ".x"
        name = "%s[%s + %d]" % (name[0], idx, tok & 0x7FF)
    mod = (tok >> 24) & 0xF
    return SRC_MODS.get(mod, "{}") .format(name) + _swizzle(tok)


# D3DXSHADER_CONSTANTTABLE RegisterSet values -> register prefix
REGSET_PREFIX = {0: "b", 1: "i", 2: "c", 3: "s"}
REGSET_SAMPLER = 3


def _ctab_info(comment_words):
    """Parse a CTAB comment block -> (target, creator, constants) where
    constants is a list of (regset, regindex, regcount, name); None if the
    block is not a CTAB."""
    if not comment_words or comment_words[0] != 0x42415443:  # 'CTAB'
        return None
    table = struct.pack("<%dI" % (len(comment_words) - 1), *comment_words[1:])

    def cstr(off):
        if off <= 0 or off >= len(table):
            return "?"
        end = table.index(b"\0", off)
        return table[off:end].decode("latin-1")

    try:
        size, creator, version, n_const, const_info, flags, target = \
            struct.unpack_from("<7I", table, 0)
    except struct.error:
        return None

    consts = []
    for i in range(n_const):
        try:
            name_off, regset, regidx, regcount, _res, _ti, _dv = \
                struct.unpack_from("<IHHHHII", table, const_info + i * 20)
        except struct.error:
            break
        consts.append((regset, regidx, regcount, cstr(name_off)))
    return cstr(target), cstr(creator), consts


def _parse_ctab(comment_words):
    """Parse a CTAB comment block -> listing lines, or None if not CTAB."""
    info = _ctab_info(comment_words)
    if info is None:
        return None
    target, creator, consts = info
    lines = ["// Constant table (CTAB):",
             "//   target:  %s" % target,
             "//   creator: %s" % creator]
    rows = []
    for regset, regidx, regcount, name in consts:
        prefix = REGSET_PREFIX.get(regset, "?")
        regs = "%s%d" % (prefix, regidx)
        if regcount > 1:
            regs += "-%s%d" % (prefix, regidx + regcount - 1)
        rows.append((prefix, regidx, "//   %-12s %s" % (regs, name)))
    for _, _, line in sorted(rows):
        lines.append(line)
    return lines


def _comment_blocks(blob):
    """Yield the word lists of every comment block in a shader blob."""
    words = struct.unpack("<%dI" % (len(blob) // 4), blob[:len(blob) // 4 * 4])
    pos = 1  # skip version token
    while pos < len(words):
        tok = words[pos]
        pos += 1
        if tok == 0x0000FFFF:
            break
        opcode = tok & 0xFFFF
        if opcode == 0xFFFE:
            length = (tok >> 16) & 0x7FFF
            yield list(words[pos:pos + length])
            pos += length
        elif opcode != 0xFFFD:
            pos += (tok >> 24) & 0xF  # skip parameter tokens (SM2+)


def sampler_names(blob):
    """{sampler_index: uniform_name} from the blob's CTAB, {} if unavailable."""
    try:
        for block in _comment_blocks(blob):
            info = _ctab_info(block)
            if info is None:
                continue
            return {regidx: name
                    for regset, regidx, regcount, name in info[2]
                    if regset == REGSET_SAMPLER}
    except Exception:  # noqa: BLE001 - best-effort metadata only
        pass
    return {}


def disassemble(blob):
    try:
        return _disassemble(blob)
    except Exception as e:  # noqa: BLE001 - must never block the UI
        return "// Disassembly failed: %s\n\n%s" % (e, _hexdump(blob))


def _disassemble(blob):
    if len(blob) < 4:
        return "// (empty shader)"
    s = _Stream(blob)

    ver_tok = s.next()
    kind = (ver_tok >> 16) & 0xFFFF
    major, minor = (ver_tok >> 8) & 0xFF, ver_tok & 0xFF
    if kind == 0xFFFE:
        is_ps = False
    elif kind == 0xFFFF:
        is_ps = True
    else:
        return "// Not D3D9 shader bytecode (first token %08x)\n\n%s" % (
            ver_tok, _hexdump(blob))
    version = (major, minor)

    header = []
    lines = ["%s_%d_%d" % ("ps" if is_ps else "vs", major, minor)]
    indent = 0

    while not s.eof():
        tok = s.next()
        if tok == 0x0000FFFF:  # end token
            break
        opcode = tok & 0xFFFF

        if opcode == 0xFFFE:  # comment block
            length = (tok >> 16) & 0x7FFF
            words = [s.next() for _ in range(min(length, len(s.words) - s.pos))]
            ctab = _parse_ctab(words)
            if ctab:
                header = ctab
            continue
        if opcode == 0xFFFD:  # ps_1_4 phase marker
            lines.append("phase")
            continue

        info = OPCODES.get(opcode)
        if info is None:
            lines.append("  // unknown opcode %#06x (token %08x)" % (opcode, tok))
            # Skip this instruction's parameter tokens (SM2+ length field)
            length = (tok >> 24) & 0xF
            for _ in range(length):
                if not s.eof():
                    s.next()
            continue

        name, ndst, nsrc = info
        control = (tok >> 16) & 0xFF

        # Predicate / opcode-specific suffixes
        if opcode in (0x29, 0x2D, 0x5E):  # ifc / breakc / setp
            name += COMPARISONS.get(control & 0x7, "")
        elif opcode == 0x42 and version >= (2, 0):  # texld variants
            if control & 0x1:
                name = "texldp"
            elif control & 0x2:
                name = "texldb"

        # def / defi / defb / dcl have literal operands
        if opcode == 0x51:  # def c#, f, f, f, f
            dst = s.next()
            vals = struct.unpack("<4f", struct.pack("<4I", *(s.next() for _ in range(4))))
            dname, _, _ = _fmt_dest(dst, is_ps, version)
            lines.append("  def %s, %g, %g, %g, %g" % ((dname,) + vals))
            continue
        if opcode == 0x30:  # defi
            dst = s.next()
            vals = struct.unpack("<4i", struct.pack("<4I", *(s.next() for _ in range(4))))
            dname, _, _ = _fmt_dest(dst, is_ps, version)
            lines.append("  defi %s, %d, %d, %d, %d" % ((dname,) + vals))
            continue
        if opcode == 0x2F:  # defb
            dst = s.next()
            val = s.next()
            dname, _, _ = _fmt_dest(dst, is_ps, version)
            lines.append("  defb %s, %s" % (dname, "true" if val else "false"))
            continue
        if opcode == 0x1F:  # dcl
            usage_tok = s.next()
            dst = s.next()
            dname, dsuffix, dmask = _fmt_dest(dst, is_ps, version)
            rtype = _reg_type(dst)
            if rtype == RT_SAMPLER:
                suffix = SAMPLER_TYPES.get((usage_tok >> 27) & 0xF, "")
            else:
                usage = usage_tok & 0x1F
                index = (usage_tok >> 16) & 0xF
                uname = USAGE_NAMES[usage] if usage < len(USAGE_NAMES) else "usage%d" % usage
                suffix = "_%s%s" % (uname, str(index) if index else "")
                # Pre-SM3 input/texcoord dcls in PS have no usage semantics
                if is_ps and version < (3, 0) and rtype in (RT_INPUT, RT_ADDR):
                    suffix = ""
            lines.append("  dcl%s%s %s%s" % (suffix, dsuffix, dname, dmask))
            continue

        # Generic instruction
        args = []
        suffix = ""
        if ndst:
            dst = s.next()
            dname, dsuffix, dmask = _fmt_dest(dst, is_ps, version)
            suffix = dsuffix
            args.append(dname + dmask)
        for _ in range(nsrc):
            src = s.next()
            rel = s.next() if (src >> 13) & 1 and version >= (2, 0) else None
            args.append(_fmt_source(src, rel, is_ps, version))

        if name in ("else", "endif", "endloop", "endrep"):
            indent = max(0, indent - 1)
        pad = "  " * (indent + 1)
        lines.append(pad + name + suffix + (" " + ", ".join(args) if args else ""))
        if name in ("if", "if_gt", "if_eq", "if_ge", "if_lt", "if_ne", "if_le",
                    "loop", "rep", "else"):
            indent += 1

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
