"""Secure manual-PDF source links.

A manual citation references a file by its **bare filename** (the ``source``
field of the chunk metadata, e.g. ``20_XPANDER_RU1.pdf``). The browser is
never given a filesystem path: it receives a stable application route

    /api/rag/sources/manual/<url-encoded filename>#page=<N>

and the backend maps the filename back to a file strictly inside the
configured ``RAG_MANUAL_PATH`` directory.

Security rules (enforced both when building a URL and when serving a file):

* only the basename is accepted — no ``/``, ``\\`` or ``..``;
* hidden files are rejected;
* only ``.pdf`` files are served;
* the resolved absolute path must stay directly inside the manual root;
* a file that does not exist yields no URL (never a broken/guessed link).
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

# The single source of truth for the route prefix. The resolver and the HTTP
# endpoint both import it so they can never drift apart.
MANUAL_SOURCE_ROUTE = "/api/rag/sources/manual"

_ALLOWED_SUFFIXES = (".pdf",)


def is_safe_manual_filename(name: str) -> bool:
    """Return True when *name* is a bare, non-hidden filename."""
    if not name or name != name.strip():
        return False
    if name in (".", ".."):
        return False
    if "/" in name or "\\" in name:
        return False
    if ".." in name:
        return False
    if name.startswith("."):
        return False
    return True


def resolve_manual_path(root: str | Path, filename: str) -> Path | None:
    """Resolve *filename* to a real file inside *root*, or None.

    Returns the resolved :class:`Path` only when every security rule holds.
    This is used by the HTTP endpoint to serve the PDF and by the resolver to
    decide whether a link can exist at all.
    """
    if not is_safe_manual_filename(filename):
        return None
    try:
        root_resolved = Path(root).resolve()
        target = (root_resolved / filename).resolve()
    except OSError:
        return None
    # ``target.parent == root_resolved`` forbids any traversal / subdirectory.
    if target.parent != root_resolved:
        return None
    if target.suffix.lower() not in _ALLOWED_SUFFIXES:
        return None
    if not target.is_file():
        return None
    return target


def manual_source_url(
    filename: str,
    *,
    page: int | None = None,
    page_from: int | None = None,
    route: str = MANUAL_SOURCE_ROUTE,
) -> str:
    """Build the browser URL for a manual PDF (optionally a page fragment).

    The filename is URL-encoded, so Cyrillic filenames such as
    ``Инструкция_по_установке_защиты_шериф.pdf`` work without exposing any
    filesystem path.
    """
    encoded = quote(filename, safe="")
    url = f"{route}/{encoded}"
    page_value = page if page is not None else page_from
    if page_value is not None:
        try:
            url += f"#page={int(page_value)}"
        except (TypeError, ValueError):
            pass
    return url
