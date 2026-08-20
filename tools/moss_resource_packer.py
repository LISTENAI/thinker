#!/usr/bin/env python3
"""Pack a generated MOSS model reference and weights into one Thinker resource."""

import argparse
import re
import struct
from pathlib import Path
from typing import List, Optional, Tuple

LABEL = b"thinker_moss10".ljust(16, b"\0")
MAGIC = 0x53534F4D
ABI_VERSION = 1
HEADER_FMT = "<16s13I"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
SECTION_ALIGN = 16
WEIGHT_ALIGN = 64
DEFAULT_SEARCH_DIRS = ("moss_res", "src/moss_demo")


def align_section(value: int, alignment: int = SECTION_ALIGN) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def append_section(
    buf: bytearray, data: bytes, alignment: int = SECTION_ALIGN
) -> Tuple[int, int]:
    offset = align_section(len(buf), alignment)
    if len(buf) < offset:
        buf.extend(b"\0" * (offset - len(buf)))
    buf.extend(data)
    return offset, len(data)


def read_weight(path: Path) -> bytes:
    if path.suffix.lower() not in {".h", ".hpp"}:
        return path.read_bytes()

    text = path.read_text(errors="ignore")
    match = re.search(r"\{(?P<body>.*?)\}\s*;", text, re.S)
    if not match:
        raise ValueError(f"cannot find weight array initializer in {path}")
    values = re.findall(r"0x[0-9a-fA-F]+|\b\d+\b", match.group("body"))
    return bytes(int(v, 0) & 0xFF for v in values)


def infer_getter(host_c: Path) -> str:
    text = host_c.read_text(errors="ignore")
    names = re.findall(r"\b(mGetModel_[A-Za-z0-9_]+)\s*\(", text)
    if not names:
        raise ValueError("--getter is required because no mGetModel_* symbol was found")
    return names[-1]


def first_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def model_file_candidates(
    model: str, search_dirs: List[Path], names: List[str]
) -> List[Path]:
    candidates = []
    seen = set()
    for directory in search_dirs:
        for base in (directory, directory / model):
            for name in names:
                path = base / name
                key = str(path)
                if key not in seen:
                    seen.add(key)
                    candidates.append(path)
    return candidates


def remove_prefix(value: str, prefix: str) -> str:
    if value.startswith(prefix):
        return value[len(prefix):]
    return value


def infer_host_c(model: str, search_dirs: List[Path]) -> Path:
    candidates = model_file_candidates(model, search_dirs, [f"{model}_host.c"])
    path = first_existing(candidates)
    if path is None:
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise ValueError(f"cannot infer --host-c for {model}; searched: {searched}")
    return path


def infer_weight(model: str, search_dirs: List[Path]) -> Path:
    names = [
        f"PackedWeights_{model}.bin",
        f"PackedWeights_{model}.h",
        f"PackedWeights_{model}.hpp",
        f"{model}_weights.bin",
        f"{model}_weight.bin",
    ]
    candidates = model_file_candidates(model, search_dirs, names)
    path = first_existing(candidates)
    if path is None:
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise ValueError(f"cannot infer --weight for {model}; searched: {searched}")
    return path


def infer_output(model: str, host_c: Path, out_dir: Optional[Path]) -> Path:
    directory = out_dir if out_dir is not None else host_c.parent
    return directory / f"{model}_moss.bin"


def normalize_target_arch(platform: str) -> str:
    platform = platform.lower()
    if platform.endswith("_cpu"):
        return platform
    return f"{platform}_cpu"


def write_registry(path: Path, getter: str, model_name: str) -> None:
    path.write_text(
        f'''#include <string.h>\n\n'''
        f'''extern void *{getter}(void);\n\n'''
        f'''void *thinker_moss_get_registered_model(const char *getter_name,\n'''
        f'''                                        const char *model_name) {{\n'''
        f'''  if ((getter_name != 0 && strcmp(getter_name, "{getter}") == 0) ||\n'''
        f'''      (model_name != 0 && strcmp(model_name, "{model_name}") == 0)) {{\n'''
        f'''    return {getter}();\n'''
        f'''  }}\n'''
        f'''  return 0;\n'''
        f'''}}\n'''
    )


def pack(args) -> None:
    if args.model is None and (args.host_c is None or args.weight is None):
        raise ValueError("provide a model name, or provide both --host-c and --weight")

    search_dirs = [path.resolve() for path in args.search_dir]
    model_hint = args.model
    if args.host_c is None:
        args.host_c = infer_host_c(model_hint, search_dirs)
    if args.weight is None:
        args.weight = infer_weight(model_hint, search_dirs)

    getter = args.getter or infer_getter(args.host_c)
    model_name = args.model_name or model_hint or remove_prefix(getter, "mGetModel_")
    target_arch = args.target_arch or normalize_target_arch(args.platform)
    if args.output is None:
        args.output = infer_output(model_name, args.host_c, args.out_dir)

    weight_data = read_weight(args.weight)

    buf = bytearray(b"\0" * align_section(HEADER_SIZE))
    weight_offset, weight_size = append_section(buf, weight_data, WEIGHT_ALIGN)
    getter_offset, getter_size = append_section(buf, getter.encode() + b"\0")
    name_offset, name_size = append_section(buf, model_name.encode() + b"\0")
    arch_offset, arch_size = append_section(buf, target_arch.encode() + b"\0")
    total_size = align_section(len(buf))
    if len(buf) < total_size:
        buf.extend(b"\0" * (total_size - len(buf)))

    header = struct.pack(
        HEADER_FMT,
        LABEL,
        MAGIC,
        ABI_VERSION,
        align_section(HEADER_SIZE),
        total_size,
        0,
        weight_offset,
        weight_size,
        getter_offset,
        getter_size,
        name_offset,
        name_size,
        arch_offset,
        arch_size,
    )
    buf[: len(header)] = header
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(buf)
    if args.registry_c:
        args.registry_c.parent.mkdir(parents=True, exist_ok=True)
        write_registry(args.registry_c, getter, model_name)
    print(
        f"packed {args.output}: total={len(buf)} weight={weight_size} "
        f"host={args.host_c} weight_file={args.weight} getter={getter} "
        f"model={model_name} arch={target_arch}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Examples:\n"
            "  python tools/moss_resource_packer.py face_keypoint\n"
            "  python tools/moss_resource_packer.py anyreid\n"
            "  python tools/moss_resource_packer.py --host-c foo_host.c --weight PackedWeights_foo.bin -o foo_moss.bin"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model", nargs="?", help="model name used to infer *_host.c, PackedWeights_* and *_moss.bin")
    parser.add_argument("--host-c", type=Path, help="generated MOSS host.c; inferred from model when omitted")
    parser.add_argument("--weight", type=Path, help="weight .bin or generated weight .h; inferred from model when omitted")
    parser.add_argument("--output", "-o", type=Path, help="output Thinker MOSS resource; defaults to <model>_moss.bin beside host.c")
    parser.add_argument("--out-dir", type=Path, help="directory for inferred output path")
    parser.add_argument(
        "--search-dir",
        action="append",
        type=Path,
        default=[Path(path) for path in DEFAULT_SEARCH_DIRS],
        help="directory searched for inferred inputs; can be repeated",
    )
    parser.add_argument("--getter", help="generated mGetModel_* symbol")
    parser.add_argument("--model-name", help="logical model name stored in the resource")
    parser.add_argument("--platform", default="venusa", help="platform name used to default --target-arch")
    parser.add_argument("--target-arch", help="target arch string, e.g. venusa_cpu")
    parser.add_argument("--registry-c", type=Path, help="optional C resolver source for static-linking this host.c")
    args = parser.parse_args()

    if args.host_c:
        args.host_c = args.host_c.resolve()
    if args.weight:
        args.weight = args.weight.resolve()
    if args.output:
        args.output = args.output.resolve()
    if args.out_dir:
        args.out_dir = args.out_dir.resolve()
    if args.registry_c:
        args.registry_c = args.registry_c.resolve()
    try:
        pack(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
