"""Diesel engine Idstring hashing (Bob Jenkins lookup8) and hashlist resolution.

Port of payday2-shader-tool's Lookup8.java / HashList.kt.

Hashing ~1.3M hashlist lines in pure Python takes several seconds, so HashList
keeps a binary cache (<hashlist>.cache) of sorted hash/name arrays; later runs
load it in well under a second and resolve names by binary search.
"""

import os
import struct
import sys
from array import array
from bisect import bisect_left

CACHE_MAGIC = b"PD2HCACH"
CACHE_VERSION = 1

MASK64 = 0xFFFFFFFFFFFFFFFF


def _mix(a, b, c):
    a = (a - b - c) & MASK64; a ^= c >> 43
    b = (b - c - a) & MASK64; b ^= (a << 9) & MASK64
    c = (c - a - b) & MASK64; c ^= b >> 8
    a = (a - b - c) & MASK64; a ^= c >> 38
    b = (b - c - a) & MASK64; b ^= (a << 23) & MASK64
    c = (c - a - b) & MASK64; c ^= b >> 5
    a = (a - b - c) & MASK64; a ^= c >> 35
    b = (b - c - a) & MASK64; b ^= (a << 49) & MASK64
    c = (c - a - b) & MASK64; c ^= b >> 11
    a = (a - b - c) & MASK64; a ^= c >> 12
    b = (b - c - a) & MASK64; b ^= (a << 18) & MASK64
    c = (c - a - b) & MASK64; c ^= b >> 22
    return a, b, c


def lookup8(data, level=0):
    a = b = level & MASK64
    c = 0x9E3779B97F4A7C13  # the golden ratio; an arbitrary value

    pos = 0
    while len(data) - pos >= 24:
        qa, qb, qc = struct.unpack_from("<QQQ", data, pos)
        a = (a + qa) & MASK64
        b = (b + qb) & MASK64
        c = (c + qc) & MASK64
        a, b, c = _mix(a, b, c)
        pos += 24

    c = (c + len(data)) & MASK64
    rest = data[pos:]
    n = len(rest)
    # The first byte of c is reserved for the length
    for i in range(n - 1, 15, -1):  # bytes 16..22 -> c
        c = (c + (rest[i] << (8 * (i - 15)))) & MASK64
    for i in range(min(n, 16) - 1, 7, -1):  # bytes 8..15 -> b
        b = (b + (rest[i] << (8 * (i - 8)))) & MASK64
    for i in range(min(n, 8) - 1, -1, -1):  # bytes 0..7 -> a
        a = (a + (rest[i] << (8 * i))) & MASK64
    a, b, c = _mix(a, b, c)
    return c


def diesel_hash(s):
    return lookup8(s.encode("utf-8"))


class HashList:
    """Maps Diesel hashes back to strings, loaded from a plain-text hashlist
    file (one string per line) via a binary cache."""

    def __init__(self):
        self._hashes = array("Q")   # sorted ascending
        self._offsets = array("I")  # parallel: offset of the name in _blob
        self._blob = b""            # NUL-terminated names
        self.path = None

    def load(self, path):
        if not self._load_cache(path):
            self._build(path)
            self._save_cache(path)
        self.path = path

    # --- lookups ---

    def get(self, hash_value):
        """Resolved string, or None if unknown."""
        i = bisect_left(self._hashes, hash_value)
        if i == len(self._hashes) or self._hashes[i] != hash_value:
            return None
        off = self._offsets[i]
        return self._blob[off:self._blob.index(b"\0", off)].decode("utf-8")

    def name_of(self, hash_value):
        """Resolved string, or the hex form if unknown."""
        return self.get(hash_value) or format(hash_value, "016x")

    def __len__(self):
        return len(self._hashes)

    # --- building and caching ---

    def _build(self, path):
        print("Building hashlist cache (one-time, takes a few seconds)...")
        entries = {}
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if line:
                    entries[diesel_hash(line)] = line

        hashes = array("Q")
        offsets = array("I")
        blob = bytearray()
        for h in sorted(entries):
            hashes.append(h)
            offsets.append(len(blob))
            blob += entries[h].encode("utf-8") + b"\0"
        self._hashes, self._offsets, self._blob = hashes, offsets, bytes(blob)

    @staticmethod
    def _cache_path(path):
        return os.path.splitext(path)[0] + ".cache"

    @staticmethod
    def _src_stamp(path):
        st = os.stat(path)
        return st.st_size, st.st_mtime_ns

    def _load_cache(self, path):
        try:
            with open(self._cache_path(path), "rb") as f:
                header = f.read(8 + 4 + 8 + 8 + 4 + 8)
                magic, version, src_size, src_mtime, count, blob_len = \
                    struct.unpack("<8sIQQIQ", header)
                if magic != CACHE_MAGIC or version != CACHE_VERSION:
                    return False
                if (src_size, src_mtime) != self._src_stamp(path):
                    return False  # hashlist changed; rebuild
                hashes = array("Q")
                offsets = array("I")
                hashes.fromfile(f, count)
                offsets.fromfile(f, count)
                blob = f.read(blob_len)
                if len(blob) != blob_len:
                    return False
        except (OSError, EOFError, struct.error, ValueError):
            return False
        if sys.byteorder == "big":
            hashes.byteswap()
            offsets.byteswap()
        self._hashes, self._offsets, self._blob = hashes, offsets, blob
        return True

    def _save_cache(self, path):
        hashes, offsets = self._hashes, self._offsets
        if sys.byteorder == "big":
            hashes, offsets = array("Q", hashes), array("I", offsets)
            hashes.byteswap()
            offsets.byteswap()
        src_size, src_mtime = self._src_stamp(path)
        tmp = self._cache_path(path) + ".tmp"
        try:
            with open(tmp, "wb") as f:
                f.write(struct.pack("<8sIQQIQ", CACHE_MAGIC, CACHE_VERSION,
                                    src_size, src_mtime, len(hashes),
                                    len(self._blob)))
                hashes.tofile(f)
                offsets.tofile(f)
                f.write(self._blob)
            os.replace(tmp, self._cache_path(path))
        except OSError:
            try:
                os.remove(tmp)
            except OSError:
                pass
