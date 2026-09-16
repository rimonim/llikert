import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture
def fixtures() -> pathlib.Path:
    return REPO / "tests" / "fixtures"
