"""CLI entry point for cents.

This module re-exports from the cli package for backwards compatibility.
The actual implementation is in the cli/ package.
"""

from cents.cli import cli, main

# Re-export for backwards compatibility
__all__ = ["cli", "main"]

if __name__ == "__main__":
    main()
