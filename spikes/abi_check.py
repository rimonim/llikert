"""M0 spike: compare ctypes struct layouts (llikert's and the binding's) with the C probe."""

import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import native  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
c_layout = json.loads((ROOT / "spikes/out/abi-c-v0.4.1.json").read_text())


def compare(label: str, py_layout: dict) -> int:
    problems = 0
    for struct, c in c_layout.items():
        py = py_layout.get(struct)
        if py is None:
            print(f"[{label}] {struct}: missing")
            problems += 1
            continue
        if py["sizeof"] != c["sizeof"]:
            print(f"[{label}] {struct}: sizeof {py['sizeof']} != C {c['sizeof']}")
            problems += 1
        for field, off in c["fields"].items():
            if py["fields"].get(field) != off:
                print(f"[{label}] {struct}.{field}: offset {py['fields'].get(field)} != C {off}")
                problems += 1
    print(f"[{label}] {problems} layout problem(s)")
    return problems


ours = compare("llikert", native.ctypes_offsets())

import llama_cpp.llama_cpp as binding  # noqa: E402

binding_layout = {}
for name in c_layout:
    s = getattr(binding, name)
    binding_layout[name] = {
        "sizeof": __import__("ctypes").sizeof(s),
        "fields": {f: getattr(s, f).offset for f, *_ in s._fields_},
    }
compare("llama-cpp-python 0.3.35", binding_layout)

header = (ROOT / "build/llama-cpp-python/vendor/llama.cpp/include/llama.h").read_text()
missing = [n for n in native.PROTOTYPES if not re.search(rf"\b{n}\s*\(", header)]
print("functions missing from llama.h:", missing or "none")
lib = native.load()
print("all declared symbols resolved in libllama:", all(hasattr(lib, n) for n in native.PROTOTYPES))
sys.exit(1 if ours or missing else 0)
