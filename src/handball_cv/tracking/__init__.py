"""Tracking support and temporal identity management."""

__all__ = ["IdentityManager"]


def __getattr__(name: str):
    # Loaded on first use: the identity manager pulls in the team model (umap,
    # ~5 s), which a caller after `geometry_cache` alone -- a render -- never needs.
    if name == "IdentityManager":
        from .identity import IdentityManager
        return IdentityManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
