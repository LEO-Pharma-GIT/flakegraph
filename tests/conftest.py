"""Shared collection rules for the whole suite.

The chart contract modules are named ``test_helm_*`` and skip themselves when
the ``helm`` binary is absent. Marking them here from the file name, rather
than inside each module, keeps the marker and the naming convention in one
place, so ``-m "not helm"`` deselects every rendered-chart contract without
each new module having to remember to declare it.
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if item.path.name.startswith("test_helm_"):
            item.add_marker(pytest.mark.helm)
