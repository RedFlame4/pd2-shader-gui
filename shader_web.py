#!/usr/bin/env python3
"""Browser-based editor for PAYDAY 2 / Diesel engine .shaders packages.

Runs a local web server (127.0.0.1 only) and opens the UI in your browser.
Works with the stock Python 3 - no dependencies, no Tk required.

Usage:
    python3 shader_web.py [file.shaders]
    python3 shader_web.py --selftest file.shaders
"""

import json
import os
import sys
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from shader_format import (ShaderPackage, ObjShaderPass, ArgType,
                           tables_for_pass, describe_state_var,
                           parse_var_value, fourcc_or_none)
from diesel_hash import HashList, diesel_hash
import d3d9_disasm


# --- config + hashlist helpers ---

TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(TOOL_DIR, "pd2_shader_gui.json")
HASHLIST_NAME = "hashlist"


def find_hashlist():
    """Path of the hashlist file sitting next to the tool, or None."""
    path = os.path.join(TOOL_DIR, HASHLIST_NAME)
    return path if os.path.isfile(path) else None


def load_config():
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
    except OSError:
        pass


def selftest(path):
    data = open(path, "rb").read()
    sp = ShaderPackage()
    sp.load(data)
    out = sp.save()
    if out == data:
        lib = sp.find_library()
        n_passes = sum(isinstance(o, ObjShaderPass) for o in sp.objects)
        print("OK: byte-identical round trip (%d bytes, %d render templates, %d passes)"
              % (len(data), len(lib.render_templates), n_passes))
        return 0
    print("FAIL: output differs (input %d bytes, output %d bytes)" % (len(data), len(out)))
    for i, (a, b) in enumerate(zip(data, out)):
        if a != b:
            print("first difference at offset %d" % i)
            break
    return 1


class Session:
    """All mutable editor state, owned by the (single-threaded) server."""

    def __init__(self):
        self.cfg = load_config()
        self.hashlist = HashList()
        self.package = None
        self.original_bytes = None
        self.file_path = None
        self.dirty_passes = set()  # ref_ids
        path = find_hashlist()
        if path:
            try:
                self.hashlist.load(path)
            except OSError:
                pass

    @property
    def dirty(self):
        return bool(self.dirty_passes)

    def open(self, path):
        path = os.path.expanduser(path)
        data = open(path, "rb").read()
        pkg = ShaderPackage()
        pkg.load(data)
        pkg.find_library()
        self.package = pkg
        self.original_bytes = data
        self.file_path = path
        self.dirty_passes = set()
        self.cfg["last_file"] = path
        save_config(self.cfg)

    def save(self, path=None):
        path = os.path.expanduser(path) if path else self.file_path
        out = self.package.save()
        check = ShaderPackage()
        check.load(out)  # never write something we can't read back
        with open(path, "wb") as f:
            f.write(out)
        self.file_path = path
        self.original_bytes = out
        self.dirty_passes = set()
        return path, len(out)

    def name_of(self, h):
        return self.hashlist.name_of(h)

    def find_pass(self, ref_id):
        for obj in self.package.objects:
            if isinstance(obj, ObjShaderPass) and obj.hdr.ref_id == ref_id:
                return obj
        raise ValueError("No shader pass with refId %d" % ref_id)

    # --- JSON payload builders ---

    def state(self):
        if not self.package:
            return {"file": None, "dirty": False,
                    "last_file": self.cfg.get("last_file"),
                    "hashlist": self._hashlist_info(), "tree": []}
        lib = self.package.find_library()
        tree = []
        by_name = sorted(lib.render_templates.items(),
                         key=lambda kv: self.name_of(kv[0]).lower())
        for h, shader in by_name:
            modes = []
            for mode_hash, passes in shader.shader_packs:
                modes.append({
                    "name": self.name_of(mode_hash),
                    "passes": [{
                        "ref": p.hdr.ref_id,
                        "label": "pass %d (vs %dB / ps %dB)" % (
                            i, len(p.vertex_shader), len(p.fragment_shader)),
                        "dirty": p.hdr.ref_id in self.dirty_passes,
                    } for i, p in enumerate(passes)],
                })
            tree.append({"name": self.name_of(h), "modes": modes})
        n_passes = sum(isinstance(o, ObjShaderPass) for o in self.package.objects)
        return {"file": self.file_path, "dirty": self.dirty,
                "hashlist": self._hashlist_info(), "tree": tree,
                "counts": {"templates": len(lib.render_templates),
                           "passes": n_passes}}

    def _hashlist_info(self):
        return {"path": self.hashlist.path, "count": len(self.hashlist)}

    def _var_json(self, sv, sdef, table):
        name, value = describe_state_var(sv, table)
        out = {"id": sv.id, "name": name, "value": value}
        if sv.flag != 0:
            out["kind"] = "idstring"
            out["edit"] = "%016x" % sv.val8
        elif sdef and sdef.arg_type == ArgType.ENUM:
            out["kind"] = "enum"
            out["choices"] = [v for v in sdef.values if v is not None]
            out["edit"] = sdef.decode(sv)
        elif sdef and sdef.arg_type == ArgType.BOOL:
            out["kind"] = "enum"
            out["choices"] = ["true", "false"]
            out["edit"] = sdef.decode(sv)
        elif sdef and sdef.arg_type == ArgType.BITMASK:
            out["kind"] = "bitmask"
            out["choices"] = list(sdef.values)
            out["edit"] = sdef.decode(sv)
        elif sdef and sdef.arg_type == ArgType.FLOAT and \
                fourcc_or_none(sv.val4) is None and \
                (fval := sdef.decode(sv)) is not None:
            out["kind"] = "float"
            out["edit"] = fval
        else:
            out["kind"] = "raw"
            fcc = fourcc_or_none(sv.val4)
            out["edit"] = fcc if fcc else "0x%08x" % sv.val4
        return out

    def pass_detail(self, ref_id):
        p = self.find_pass(ref_id)
        sv_table, tex_table = tables_for_pass(p)
        state_vars = [self._var_json(sv, sv_table.get(sv.id), sv_table)
                      for sv in p.state_vars]
        ps_samplers = d3d9_disasm.sampler_names(p.fragment_shader)
        vs_samplers = d3d9_disasm.sampler_names(p.vertex_shader)
        textures = []
        for block in p.textures:
            if block.ukn_i in ps_samplers:
                label = "s%d — %s" % (block.ukn_i, ps_samplers[block.ukn_i])
            elif block.ukn_i in vs_samplers:
                label = "s%d — %s (vertex)" % (block.ukn_i, vs_samplers[block.ukn_i])
            else:
                label = "sampler %d" % block.ukn_i
            textures.append({
                "label": label,
                "vars": [self._var_json(sv, tex_table.get(sv.id), tex_table)
                         for sv in block.vars],
            })
        return {
            "ref": ref_id,
            "state_vars": state_vars,
            "textures": textures,
            "vs": {"size": len(p.vertex_shader),
                   "asm": d3d9_disasm.disassemble(p.vertex_shader)},
            "ps": {"size": len(p.fragment_shader),
                   "asm": d3d9_disasm.disassemble(p.fragment_shader)},
        }

    def set_var(self, ref_id, scope, tex_index, var_index, text):
        p = self.find_pass(ref_id)
        sv_table, tex_table = tables_for_pass(p)
        if scope == "sv":
            sv = p.state_vars[var_index]
            sdef = sv_table.get(sv.id)
        else:
            sv = p.textures[tex_index].vars[var_index]
            sdef = tex_table.get(sv.id)
        value = parse_var_value(sv, sdef, text, diesel_hash)
        if sv.flag != 0:
            sv.val8 = value
        else:
            sv.val4 = value
        self.dirty_passes.add(ref_id)

    def set_blob(self, ref_id, which, blob):
        p = self.find_pass(ref_id)
        if which == "vs":
            p.vertex_shader = blob
        else:
            p.fragment_shader = blob
        self.dirty_passes.add(ref_id)

    def verify(self):
        out = self.package.save()
        if not self.dirty:
            ok = out == self.original_bytes
            return {"ok": ok, "message":
                    "Byte-identical to the file on disk (%d bytes)." % len(out)
                    if ok else "Output differs from the original file!"}
        check = ShaderPackage()
        check.load(out)
        ok = check.save() == out
        return {"ok": ok, "message":
                "File has edits; output reparses cleanly and is self-consistent "
                "(%d bytes)." % len(out) if ok else "Output is not self-consistent!"}

    def export_all(self, out_dir):
        out_dir = os.path.expanduser(out_dir)
        lib = self.package.find_library()
        count = 0
        for h, shader in sorted(lib.render_templates.items()):
            t_name = safe_name(self.name_of(h))
            for mode_hash, passes in shader.shader_packs:
                m_name = safe_name(self.name_of(mode_hash))
                for i, p in enumerate(passes):
                    d = os.path.join(out_dir, t_name, m_name)
                    os.makedirs(d, exist_ok=True)
                    for suffix, blob in (("vsb", p.vertex_shader),
                                         ("psb", p.fragment_shader)):
                        with open(os.path.join(d, "pass%d.%s" % (i, suffix)), "wb") as f:
                            f.write(blob)
                        with open(os.path.join(d, "pass%d.%s.asm" % (i, suffix)), "w") as f:
                            f.write(d3d9_disasm.disassemble(blob))
                        count += 1
        return count


def safe_name(name, max_len=120):
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    return safe[:max_len]


SESSION = Session()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep the terminal quiet

    # --- helpers ---

    def _send_bytes(self, body, ctype, status=200, headers=()):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status=200):
        self._send_bytes(json.dumps(obj).encode("utf-8"), "application/json", status)

    def _error(self, e, status=400):
        self._json({"error": str(e)}, status)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(n)

    def _body_json(self):
        return json.loads(self._body() or b"{}")

    _STATIC_TYPES = {".css": "text/css", ".js": "application/javascript"}

    def _static(self, name):
        # Serve vendored assets (Pico CSS, etc.) from the script's own static/ dir.
        path = os.path.join(TOOL_DIR, "static", name)
        if os.sep in name or not os.path.isfile(path):
            self._error("Not found", 404)
            return
        ctype = self._STATIC_TYPES.get(os.path.splitext(name)[1],
                                       "application/octet-stream")
        with open(path, "rb") as f:
            data = f.read()
        self._send_bytes(data, ctype,
                         headers=[("Cache-Control", "max-age=86400")])

    # --- routes ---

    def do_GET(self):
        url = urlparse(self.path)
        try:
            if url.path == "/":
                self._send_bytes(read_page(), "text/html; charset=utf-8")
            elif url.path.startswith("/static/"):
                self._static(os.path.basename(url.path))
            elif url.path == "/api/state":
                self._json(SESSION.state())
            elif url.path.startswith("/api/pass/"):
                self._json(SESSION.pass_detail(int(url.path.rsplit("/", 1)[1])))
            elif url.path == "/api/blob":
                q = parse_qs(url.query)
                p = SESSION.find_pass(int(q["ref"][0]))
                which = q["which"][0]
                blob = p.vertex_shader if which == "vs" else p.fragment_shader
                fname = "pass%d.%s" % (p.hdr.ref_id, "vsb" if which == "vs" else "psb")
                self._send_bytes(blob, "application/octet-stream", headers=[
                    ("Content-Disposition", 'attachment; filename="%s"' % fname)])
            else:
                self._error("Not found", 404)
        except Exception as e:  # noqa: BLE001 - report to UI instead of dying
            self._error(e)

    def do_POST(self):
        url = urlparse(self.path)
        try:
            if url.path == "/api/open":
                SESSION.open(self._body_json()["path"])
                self._json(SESSION.state())
            elif url.path == "/api/save":
                path, size = SESSION.save(self._body_json().get("path"))
                self._json({"ok": True, "message": "Saved %s (%d bytes)" % (path, size),
                            "state": SESSION.state()})
            elif url.path == "/api/var":
                b = self._body_json()
                SESSION.set_var(b["ref"], b["scope"], b.get("tex", 0),
                                b["index"], b["value"])
                self._json({"ok": True, "detail": SESSION.pass_detail(b["ref"]),
                            "state": SESSION.state()})
            elif url.path == "/api/blob":
                q = parse_qs(url.query)
                ref = int(q["ref"][0])
                SESSION.set_blob(ref, q["which"][0], self._body())
                self._json({"ok": True, "detail": SESSION.pass_detail(ref),
                            "state": SESSION.state()})
            elif url.path == "/api/verify":
                self._json(SESSION.verify())
            elif url.path == "/api/export_all":
                count = SESSION.export_all(self._body_json()["dir"])
                self._json({"ok": True,
                            "message": "Wrote %d blobs (plus disassembly)." % count})
            else:
                self._error("Not found", 404)
        except Exception as e:  # noqa: BLE001
            self._error(e)


PAGE_PATH = os.path.join(TOOL_DIR, "index.html")


def read_page():
    """The UI shell, read fresh each request so edits show on reload."""
    with open(PAGE_PATH, "rb") as f:
        return f.read()


def main():
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        if len(args) != 2:
            print("usage: shader_web.py --selftest <file.shaders>")
            return 2
        return selftest(args[1])

    if args:
        SESSION.open(args[0])
        print("Loaded %s" % SESSION.file_path)

    port = 8741
    while True:
        try:
            server = HTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            port += 1
    url = "http://127.0.0.1:%d/" % port
    print("PD2 Shader Tool running at %s  (Ctrl+C to quit)" % url)
    threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
