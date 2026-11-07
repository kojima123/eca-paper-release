"""Runtime guardrails for Tier 1 strict-generic submission rehearsals."""

from __future__ import annotations

import importlib.abc
import os
import sys
from typing import Any


STRICT_GENERIC_ENV = "V2_TIER1_STRICT_GENERIC"
_ADAPTER_PREFIX = "arcgym.adapters"


class Tier1AdapterImportError(ImportError):
    """Raised when Tier 1 strict-generic mode observes adapter imports."""


class _AdapterImportBan(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: Any = None,
    ) -> None:
        if _is_adapter_module(fullname):
            raise Tier1AdapterImportError(
                f"{STRICT_GENERIC_ENV}=1 forbids adapter import: {fullname}"
            )
        return None


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name, "") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _is_adapter_module(module_name: str) -> bool:
    return module_name == _ADAPTER_PREFIX or module_name.startswith(f"{_ADAPTER_PREFIX}.")


def _loaded_adapter_modules() -> list[str]:
    return sorted(name for name in sys.modules if _is_adapter_module(name))


def tier1_strict_generic_enabled() -> bool:
    return _env_truthy(STRICT_GENERIC_ENV)


def install_tier1_guard() -> bool:
    """Install the Tier 1 adapter-import ban when requested by env.

    Returns True when strict mode is active. If an adapter has already been
    imported, fail immediately so a dry-run cannot silently continue after the
    source-derived path has entered the process.
    """

    if not tier1_strict_generic_enabled():
        return False

    loaded = _loaded_adapter_modules()
    if loaded:
        raise Tier1AdapterImportError(
            f"{STRICT_GENERIC_ENV}=1 forbids preloaded adapter modules: {loaded}"
        )

    if not any(isinstance(finder, _AdapterImportBan) for finder in sys.meta_path):
        sys.meta_path.insert(0, _AdapterImportBan())
    return True


install_tier1_guard()


__all__ = [
    "STRICT_GENERIC_ENV",
    "Tier1AdapterImportError",
    "install_tier1_guard",
    "tier1_strict_generic_enabled",
]
