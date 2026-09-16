"""Copy the shared test fixtures into the R package (R CMD check cannot see the repo root).

    python tools/sync_fixtures.py          # copy
    python tools/sync_fixtures.py --check  # fail if the R copy differs
"""

import filecmp
import pathlib
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
SOURCE = REPO / "tests" / "fixtures"
TARGET = REPO / "r" / "llikert" / "tests" / "testthat" / "fixtures"
SHARED = ["protocol", "runs", "tasks", "checkpoints"]


def files(root: pathlib.Path) -> set[pathlib.Path]:
    return {p.relative_to(root) for d in SHARED if (root / d).exists() for p in (root / d).rglob("*") if p.is_file()}


def main() -> int:
    source, target = files(SOURCE), files(TARGET)
    if "--check" in sys.argv:
        differ = sorted(str(p) for p in source ^ target) + sorted(str(p) for p in source & target if not filecmp.cmp(SOURCE / p, TARGET / p, shallow=False))
        if differ:
            print("R fixtures out of sync:", ", ".join(differ[:10]))
            return 1
        print("R fixtures in sync")
        return 0
    for d in SHARED:
        shutil.rmtree(TARGET / d, ignore_errors=True)
        if (SOURCE / d).exists():
            shutil.copytree(SOURCE / d, TARGET / d)
    print(f"copied {len(source)} fixture files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
