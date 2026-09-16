import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]


def test_r_package_fixtures_match_shared_fixtures():
    out = subprocess.run([sys.executable, str(REPO / "tools/sync_fixtures.py"), "--check"], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout
