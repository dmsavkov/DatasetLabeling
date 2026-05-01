from __future__ import annotations

from dotenv import load_dotenv

_DOTENV_LOADED = False
_DOTENV_OVERRIDDEN = False


def load_dotenv_if_present(*, override: bool = True) -> None:
    """
    Load environment variables from a local `.env` file (if present).

    This is intentionally a tiny, idempotent helper so any module that reads API
    tokens from `os.environ` can safely call it at import time.
    """

    global _DOTENV_LOADED, _DOTENV_OVERRIDDEN
    if _DOTENV_LOADED and (not override or _DOTENV_OVERRIDDEN):
        return
    load_dotenv(override=override)
    _DOTENV_LOADED = True
    _DOTENV_OVERRIDDEN = _DOTENV_OVERRIDDEN or override
