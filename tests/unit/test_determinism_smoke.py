import os


def test_pythonhashseed_is_set_for_ci_contract() -> None:
    if os.getenv("CI") == "true":
        assert os.getenv("PYTHONHASHSEED") == "0"
