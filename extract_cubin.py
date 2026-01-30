#!/usr/bin/env python3
"""Extract and disassemble cubins from TensorRT engine files."""

import io
import subprocess
import sys
import tempfile
from pathlib import Path
from elftools.elf.elffile import ELFFile


def find_cubins(data: bytes) -> list:
    """Find all ELF magic byte offsets."""
    offsets = []
    pos = 0
    while (pos := data.find(b'\x7fELF', pos)) != -1:
        offsets.append(pos)
        pos += 1
    return offsets


def get_cubin_size(data: bytes, offset: int) -> int:
    """Calculate cubin size using pyelftools."""
    elf = ELFFile(io.BytesIO(data[offset:]))
    hdr = elf.header
    # Section header table end
    max_end = hdr['e_shoff'] + hdr['e_shentsize'] * hdr['e_shnum']
    # Program header table end
    max_end = max(max_end, hdr['e_phoff'] + hdr['e_phentsize'] * hdr['e_phnum'])
    # Section data ends
    for section in elf.iter_sections():
        max_end = max(max_end, section['sh_offset'] + section['sh_size'])
    return max_end


def disassemble(cubin_data: bytes) -> str:
    """Disassemble cubin using nvdisasm."""
    with tempfile.NamedTemporaryFile(suffix='.cubin', delete=False) as f:
        f.write(cubin_data)
        path = f.name
    try:
        result = subprocess.run(['nvdisasm', path], capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    finally:
        Path(path).unlink(missing_ok=True)


def main(engine_path: str):
    print(f"Reading: {engine_path}")
    data = Path(engine_path).read_bytes()
    
    offsets = find_cubins(data)
    print(f"Found {len(offsets)} cubins\n")
    
    # Just extract and disassemble first cubin
    offset = offsets[0]
    size = get_cubin_size(data, offset)
    cubin = data[offset:offset + size]
    
    print(f"Cubin #0 @ {hex(offset)}, {size/1024:.1f} KB\n")
    print(disassemble(cubin))


if __name__ == "__main__":
    engine = sys.argv[1] if len(sys.argv) > 1 else "./qwen2.5_1.5b_engine/rank0.engine"
    main(engine)
