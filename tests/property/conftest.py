from __future__ import annotations

from hypothesis import settings
from hypothesis.errors import InvalidArgument

_PROPERTY_PROFILE = "rfmna_property_ci"


def pytest_configure(config: object) -> None:
    del config
    try:
        settings.get_profile(_PROPERTY_PROFILE)
    except InvalidArgument:
        settings.register_profile(
            _PROPERTY_PROFILE,
            settings(
                derandomize=True,
                max_examples=50,
                deadline=None,
                print_blob=True,
            ),
        )
    settings.load_profile(_PROPERTY_PROFILE)
