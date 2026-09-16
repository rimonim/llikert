import pytest

from llikert import RetryPolicy, Scorer

from .mock_service import MockService, load_fixture


class Sleeper:
    def __init__(self):
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


@pytest.fixture
def mock():
    return MockService()


@pytest.fixture
def sleeper():
    return Sleeper()


@pytest.fixture
def connect(mock, sleeper):
    def _connect(**kwargs):
        kwargs.setdefault("retry", RetryPolicy(sleep=sleeper))
        return Scorer.connect("http://mock", _client=mock.client(), **kwargs)

    return _connect


@pytest.fixture
def dataset():
    return load_fixture("runs/dataset.json")
