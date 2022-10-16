import pytest  # NOQA

from ml_hands_on.sample import hello


def test_hello() -> None:
    assert hello() == "Hello World!"
