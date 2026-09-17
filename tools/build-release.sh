#!/usr/bin/env bash
# Build release artifacts into dist/: Python sdist and wheel, R source package, checksums,
# and a manifest tying them to the source revision, lockfiles, and service image.
#
#   tools/build-release.sh [image-tag]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=${1:-llikert-service:0.1.0.dev0-cuda12.9}
DIST=$ROOT/dist
PY=${PYTHON:-$ROOT/build/venv-lock/bin/python}
rm -rf "$DIST" && mkdir -p "$DIST"

"$PY" -m pip install -q build
"$PY" -m build --outdir "$DIST" "$ROOT/python"
(cd "$DIST" && R CMD build --no-build-vignettes "$ROOT/r/llikert" >/dev/null)

(cd "$DIST" && sha256sum -- *.whl *.tar.gz > SHA256SUMS)

if ! image_id=$(docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null) || [ -z "$image_id" ]; then
  image_id="not built"
fi
revision=$(git -C "$ROOT" rev-parse HEAD)
dirty=$(git -C "$ROOT" status --porcelain | grep -q . && echo true || echo false)
"$PY" - "$DIST" "$IMAGE" "$image_id" "$revision" "$dirty" <<'PYEOF'
import hashlib, json, pathlib, sys
dist, image, image_id, revision, dirty = sys.argv[1:]
root = pathlib.Path(dist).parent
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
manifest = {
    "source_revision": revision,
    "source_tree_dirty": dirty == "true",
    "artifacts": {p.name: sha(p) for p in sorted(pathlib.Path(dist).iterdir()) if p.suffix in (".whl", ".gz")},
    "service_image": {"tag": image, "id": image_id},
    "lockfiles": {name: sha(root / "deploy" / name) for name in ("requirements-server.lock", "requirements-build.lock")},
    "pinned_stack": {"llama.cpp": "v0.4.1 (b29c606e28a01b1bc8c1351026a0fa6e616bf6c4)", "llama-cpp-python": "v0.3.35 (3691546f1c9e0c1bf93323dff02230bd959cf562)",
                     "cuda_base": "nvidia/cuda:12.9.0 (devel sha256:26cd7d00…, runtime sha256:d1ad87e9…)"},
    "supported_model": {"file": "qwen3-4b-instruct-2507-f32.gguf", "sha256": "a5733d5a25e8b824e1ef9d7e75449ffd8f8ec582de8ac253420e4423a7f0e357"},
}
(pathlib.Path(dist) / "release-manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
print(json.dumps(manifest, indent=1))
PYEOF
