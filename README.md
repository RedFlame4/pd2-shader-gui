# PD2 Shader Tool

GUI editor for PAYDAY 2 / Diesel engine `.shaders` packages (e.g.
`old_deferred_lighting_latest.shaders`). Parse, inspect, edit and repack —
no dependencies beyond the stock Python 3.

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
  decoded to readable names and values. Click a value to edit; enums become
  dropdowns, bitmasks become checkboxes.
- **Textures tab** — per-sampler settings (filtering, addressing, sRGB…),
  same click-to-edit.
- **Vertex/Pixel Shader tabs** — built-in D3D9 SM1–3 disassembly with the
  CTAB constant table (uniform names → registers) at the top. *Export blob…*
  saves the raw bytecode; *Import blob…* swaps in a replacement (e.g. compiled
  with `fxc /T ps_3_0`). Sizes are repacked automatically on save.
- **Export all blobs…** dumps every pass's bytecode plus disassembly to a
  directory tree.
- **Verify round-trip** proves the writer is lossless: an unedited file saves
  byte-identical; an edited one must reparse self-consistently.
- Save never writes a file the parser can't read back.

## Files

| File | Purpose |
|---|---|
| `shader_web.py` | Browser UI + local server (run this) |
| `shader_format.py` | `.shaders` parser/serializer + state-var tables |
| `d3d9_disasm.py` | D3D9 shader bytecode disassembler |
| `diesel_hash.py` | Diesel Idstring hash (Bob Jenkins lookup8) + hashlist |
| `static/` | Vendored web assets (Tailwind + Alpine), served locally/offline |

Settings (last file/directory) persist in `pd2_shader_gui.json` next to the tool.

## Format notes

Ported from [znix's payday2-shader-tool](https://gitlab.com/znixian/payday2-shader-tool), targeting the
**D3D (Windows)** files — like `old_deferred_lighting_latest.shaders`. Texture
blocks carry no name string, state-var IDs are raw `D3DRENDERSTATETYPE` values,
and sampler-var IDs are `D3DSAMPLERSTATETYPE` values (sampler names live in the
bytecode's CTAB instead). Packages load and save byte-identically.
