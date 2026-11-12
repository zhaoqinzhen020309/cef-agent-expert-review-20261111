"""Recover CRC-verified members when the publisher ZIP lacks a central directory.

Each recovered member must have a complete deflate stream and a matching data
descriptor. Never treat partial or unverified bytes as a benchmark instance.
"""
from pathlib import Path
import json
import struct
import zlib


def read_members(path):
    data = Path(path).read_bytes()
    offset = 0
    while data[offset:offset + 4] == b"PK\x03\x04":
        header = struct.unpack_from("<4s5H3I2H", data, offset)
        _, _, flags, method, _, _, crc, compressed_size, size, name_len, extra_len = header
        name = data[offset + 30:offset + 30 + name_len].decode("utf-8")
        start = offset + 30 + name_len + extra_len
        if flags & 1:
            raise ValueError("encrypted member")
        if method == 0 and not flags & 8:
            payload = data[start:start + compressed_size]
            offset = start + compressed_size
        elif method == 8:
            decoder = zlib.decompressobj(-15)
            payload = decoder.decompress(data[start:])
            if not decoder.eof:
                raise ValueError(f"truncated archive member: {name}")
            consumed = len(data) - start - len(decoder.unused_data)
            offset = start + consumed
            if flags & 8:
                if data[offset:offset + 4] == b"PK\x07\x08":
                    offset += 4
                crc, compressed_size, size = struct.unpack_from("<3I", data, offset)
                offset += 12
            if consumed != compressed_size:
                raise ValueError("compressed length mismatch")
        else:
            raise ValueError(f"unsupported ZIP member method/flags: {name}")
        if len(payload) != size or zlib.crc32(payload) != crc:
            raise ValueError(f"member CRC/length verification failed: {name}")
        yield name, payload


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "data/raw/fssp"
    verified = []
    for name, payload in read_members(root / "TestSolutions.zip"):
        if "/taillard/" in name and name.endswith(".dat"):
            target = root / "taillard" / Path(name).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            verified.append(name)
        if len(verified) == 120:
            break
    if len(verified) != 120:
        raise ValueError(f"expected 120 CRC-verified Taillard instances, got {len(verified)}")
    (root / "member_verification.json").write_text(json.dumps({"status": "member_crc_verified", "members": verified}, indent=2))
    print(len(verified))
