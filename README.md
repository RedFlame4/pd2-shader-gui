# PD2 Shader Tool

GUI editor for PAYDAY 2 / Diesel engine `.shaders` packages (e.g.
`old_deferred_lighting_latest.shaders`, `deferred_lighting.d3d11.shaders`).
Parse, inspect, edit and repack — no dependencies beyond the stock Python 3.
Both the D3D9 (SM1-3) and D3D11 (SM4-5, DXBC) package layouts are supported;
the layout is auto-detected per file (see Format notes below).

## Run it

macOS / Linux:

```sh
python3 shader_web.py ../old_deferred_lighting_latest.shaders
```

Windows (needs [Python 3](https://www.python.org/downloads/) installed):

```bat
py shader_web.py path\to\file.shaders
```

This starts a local server (127.0.0.1 only) and opens the editor in your
browser. Quit with Ctrl+C in the terminal.

Verify any file round-trips byte-identically (no GUI):

```sh
python3 shader_web.py --selftest file.shaders
```

## What you can do

- **Browse** render templates → modes → passes in the tree. Names resolve via
  an optional hashlist file (one string per line) named `hashlist`,
  placed in the same directory as the tool —
  it is loaded automatically on startup. Without one, names show as 16-digit
  Diesel hashes. The first start builds a `<hashlist>.cache` file (a few
  seconds); later starts load it near-instantly, and it rebuilds itself
  whenever the hashlist file changes.
- **State Vars tab** — per-pass render state (depth/stencil/blend/cull…),
  decoded to readable names and values for D3D9 packages. Click a value to
  edit; enums become dropdowns, bitmasks become checkboxes. D3D11 packages
  use a different state-var id space that has been partly reverse-engineered
  (by diffing shared passes against the D3D9 build): the common
  depth/stencil/blend/cull states decode to names, but any id not yet
  identified still shows and edits as a raw number.
- **Textures tab** — per-sampler settings (filtering, addressing, sRGB…),
  same click-to-edit. Sampler names resolve from the shader's CTAB (D3D9) or
  from the block's own Idstring hash via the hashlist (D3D11).
- **Vertex/Pixel Shader tabs** — built-in disassembly: D3D9 SM1–3 with the
  CTAB constant table (uniform names → registers) at the top, or D3D11
  SM4/5 (DXBC) with an RDEF-derived header (cbuffer layout, bound
  textures/samplers). *Export blob…* saves the raw bytecode; *Import blob…*
  swaps in a replacement (e.g. compiled with `fxc /T ps_3_0` or `ps_5_0`).
  Sizes are repacked automatically on save.
- **Export all blobs…** dumps every pass's bytecode plus disassembly to a
  directory tree.
- Save never writes a file the parser can't read back.

## Files

| File | Purpose |
|---|---|
| `shader_web.py` | Browser UI + local server (run this) |
| `shader_format.py` | `.shaders` parser/serializer + state-var tables |
| `d3d9_disasm.py` | D3D9 shader bytecode disassembler |
| `d3d11_disasm.py` | D3D11 (DXBC/SM4-5) shader bytecode disassembler |
| `diesel_hash.py` | Diesel Idstring hash (Bob Jenkins lookup8) + hashlist |
| `static/` | Vendored web assets (Tailwind + Alpine), served locally/offline |

Settings (last file/directory) persist in `pd2_shader_gui.json` next to the tool.

## Format notes

Ported from [znix's payday2-shader-tool](https://gitlab.com/znixian/payday2-shader-tool)
(D3D9-only), then extended here to also read the newer **D3D11** package
layout — like `deferred_lighting.d3d11.shaders`. Both share the same outer
container (render templates → modes → passes); the tool auto-detects a
pass's layout while loading by trying both texture-block shapes and checking
which one leaves the bytecode blobs looking like real shader bytecode
(D3D9 SM1-3 tokens or a `DXBC` container).

**D3D9 layout** (`old_deferred_lighting_latest.shaders` and similar):
texture blocks carry no name string, state-var IDs are raw
`D3DRENDERSTATETYPE` values, and sampler-var IDs are `D3DSAMPLERSTATETYPE`
values (sampler names live in the bytecode's CTAB instead).

**D3D11 layout**: each texture block is keyed by the full 8-byte Idstring
hash of the sampler/texture variable's name (resolved via the hashlist)
rather than a small register index, since D3D11 bytecode (DXBC, no CTAB)
has no equivalent table to recover names from. The per-pass state-var id
space is also different from D3D9's `D3DRENDERSTATETYPE` — D3D11 has no
`SetRenderState`-style API, so Diesel uses its own compact ids here. That
mapping has been partly reverse-engineered (by diffing shared passes against
the D3D9 build): the common depth/stencil/blend/cull states decode to
names/enums, while any id not yet identified round-trips losslessly and
displays/edits as a raw number. The per-sampler var ids are likewise their
own space, only partly recovered — `AddressU`/`AddressV` decode, but D3D11's
packed filter and the remaining sampler ids stay raw.

Packages load and save byte-identically in both layouts.
