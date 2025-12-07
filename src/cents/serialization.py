"""JSON serialization helpers for cents dataclasses."""

from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any


def serialize(obj: Any) -> Any:
    """Serialize a dataclass, enum, datetime, or nested structure to JSON-compatible types.

    Handles:
    - Dataclasses: Recursively converts to dict
    - Enums: Returns .value
    - Datetimes: Returns ISO format string
    - Lists/dicts: Recursively serializes contents
    - None and primitives: Passed through unchanged

    Args:
        obj: Any Python object to serialize

    Returns:
        JSON-serializable representation of the object

    Example:
        >>> from cents.models import Thesis
        >>> thesis = repo.get("abc123")
        >>> json.dumps(serialize(thesis))
    """
    if obj is None:
        return None
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, datetime):
        return obj.isoformat()
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    return obj
