import json
import subprocess
import sys

HEAVY = ["numpy", "fastapi", "pydantic", "jinja2", "uvicorn", "llama_cpp", "torch", "transformers"]


def test_base_import_is_lightweight():
    code = f"import json, sys, llikert; print(json.dumps([m for m in {HEAVY!r} if m in sys.modules]))"
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout
    assert json.loads(out) == []


def test_cli_help_does_not_import_server():
    code = (
        "import json, sys\n"
        "from llikert import cli\n"
        "try:\n    cli.main(['--help'])\nexcept SystemExit:\n    pass\n"
        f"print(json.dumps([m for m in {HEAVY!r} if m in sys.modules]))"
    )
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout.splitlines()[-1]
    assert json.loads(out) == []
