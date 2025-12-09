"""Dataset management for cents.

Manages named datasets (database files) allowing users to switch between
different portfolios without affecting the default database.

Datasets are stored in ~/.cents/datasets.toml:
    active = "default"

    [datasets]
    default = "~/.cents/data/cents.db"
    friend = "~/Downloads/friend-portfolio.db"
"""

from __future__ import annotations

from pathlib import Path
import tomllib

# Use tomli_w for writing TOML if available, otherwise manual formatting
try:
    import tomli_w

    _HAS_TOMLI_W = True
except ImportError:
    _HAS_TOMLI_W = False


def get_datasets_path() -> Path:
    """Get the path to the datasets config file."""
    return Path.home() / ".cents" / "datasets.toml"


def _get_default_db_path() -> Path:
    """Get the default database path."""
    return Path.home() / ".cents" / "data" / "cents.db"


def _expand_path(path_str: str) -> Path:
    """Expand ~ and resolve path."""
    return Path(path_str).expanduser().resolve()


def load_datasets() -> dict:
    """Load datasets config, creating default if needed."""
    path = get_datasets_path()

    if not path.exists():
        # Return default config structure
        return {
            "active": "default",
            "datasets": {"default": str(_get_default_db_path())},
        }

    try:
        data = tomllib.loads(path.read_text())
    except (tomllib.TOMLDecodeError, OSError):
        # If file is corrupted, return defaults
        return {
            "active": "default",
            "datasets": {"default": str(_get_default_db_path())},
        }

    # Ensure required fields exist
    if "active" not in data:
        data["active"] = "default"
    if "datasets" not in data:
        data["datasets"] = {}
    if "default" not in data["datasets"]:
        data["datasets"]["default"] = str(_get_default_db_path())

    return data


def save_datasets(data: dict) -> None:
    """Save datasets config to file."""
    path = get_datasets_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    if _HAS_TOMLI_W:
        path.write_bytes(tomli_w.dumps(data))
    else:
        # Manual TOML formatting
        lines = [f'active = "{data["active"]}"', "", "[datasets]"]
        for name, db_path in data.get("datasets", {}).items():
            lines.append(f'{name} = "{db_path}"')
        path.write_text("\n".join(lines) + "\n")


def get_active_dataset() -> tuple[str, Path]:
    """Get the currently active dataset name and path.

    Returns:
        Tuple of (name, path) for the active dataset.
    """
    data = load_datasets()
    name = data["active"]
    path_str = data["datasets"].get(name)

    if path_str is None:
        # Active dataset doesn't exist, fall back to default
        name = "default"
        path_str = data["datasets"].get("default", str(_get_default_db_path()))

    return name, _expand_path(path_str)


def set_active_dataset(name: str) -> None:
    """Switch to a named dataset.

    Args:
        name: Name of the dataset to activate.

    Raises:
        ValueError: If the dataset doesn't exist.
    """
    data = load_datasets()

    if name not in data["datasets"]:
        available = ", ".join(data["datasets"].keys())
        raise ValueError(f"Dataset '{name}' not found. Available: {available}")

    data["active"] = name
    save_datasets(data)


def add_dataset(name: str, path: str | Path) -> Path:
    """Register a new named dataset.

    Args:
        name: Name for the dataset (must be unique).
        path: Path to the database file.

    Returns:
        The resolved path to the database.

    Raises:
        ValueError: If name already exists or is invalid.
    """
    if not name or name.strip() != name:
        raise ValueError("Dataset name cannot be empty or have leading/trailing spaces")

    data = load_datasets()

    if name in data["datasets"]:
        raise ValueError(f"Dataset '{name}' already exists")

    resolved = _expand_path(str(path))
    data["datasets"][name] = str(resolved)
    save_datasets(data)

    return resolved


def remove_dataset(name: str) -> None:
    """Remove a named dataset (does not delete the database file).

    Args:
        name: Name of the dataset to remove.

    Raises:
        ValueError: If trying to remove 'default' or dataset doesn't exist.
    """
    if name == "default":
        raise ValueError("Cannot remove the 'default' dataset")

    data = load_datasets()

    if name not in data["datasets"]:
        raise ValueError(f"Dataset '{name}' not found")

    del data["datasets"][name]

    # If we removed the active dataset, switch to default
    if data["active"] == name:
        data["active"] = "default"

    save_datasets(data)


def list_datasets() -> dict[str, tuple[Path, bool]]:
    """List all registered datasets.

    Returns:
        Dict mapping name to (path, is_active) tuple.
    """
    data = load_datasets()
    active = data["active"]

    result = {}
    for name, path_str in data["datasets"].items():
        result[name] = (_expand_path(path_str), name == active)

    return result
