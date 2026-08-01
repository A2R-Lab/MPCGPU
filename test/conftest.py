import pytest


def pytest_collection_modifyitems(config, items):
    # The signed pytest-gpu-proof receipt attests the FULL gate suite —
    # every item gets the plugin's gpu_proof marker (mirrors GATO).
    for item in items:
        item.add_marker(pytest.mark.gpu_proof)
