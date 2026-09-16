import pathlib

from llikert.server.schemas import export_schemas

REPO = pathlib.Path(__file__).resolve().parents[3]


def test_committed_schemas_match_models(capsys):
    assert export_schemas(REPO / "schemas", check=True) == 0, capsys.readouterr().out
