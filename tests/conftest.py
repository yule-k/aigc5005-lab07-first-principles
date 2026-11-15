import pytest

from prepare import make_splits


@pytest.fixture(scope="session")
def splits():
    """Build the three datasets once and share them with every test."""
    return make_splits(42)
