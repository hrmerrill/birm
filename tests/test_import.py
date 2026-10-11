from importlib import import_module


def test_import():
    """The package is importable."""
    import_module("birm")
