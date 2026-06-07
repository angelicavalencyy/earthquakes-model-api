from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _parse_float(value: Any, *, field_name: str) -> float:
    if value is None:
        raise ValueError(f"{field_name} is required")

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    match = _FLOAT_RE.search(text)
    if not match:
        raise ValueError(f"{field_name} must be a number")

    return float(match.group(0))


class BMKGGempaItem(BaseModel):
    """One BMKG realtime earthquake item.

    This model validates only the fields needed by realtime inference.
    Other fields are allowed and kept as-is.
    """

    model_config = ConfigDict(extra="allow")

    Magnitude: float = Field(..., description="Earthquake magnitude")
    Kedalaman: float = Field(..., description="Earthquake depth (km)")

    @field_validator("Magnitude", mode="before")
    @classmethod
    def _validate_magnitude(cls, v: Any) -> float:
        return _parse_float(v, field_name="Magnitude")

    @field_validator("Kedalaman", mode="before")
    @classmethod
    def _validate_kedalaman(cls, v: Any) -> float:
        # BMKG commonly sends strings like "10 km".
        return _parse_float(v, field_name="Kedalaman")


class BMKGInfoGempa(BaseModel):
    model_config = ConfigDict(extra="allow")

    gempa: list[BMKGGempaItem] = Field(..., description="List of earthquake items")


class BMKGRealtimePayload(BaseModel):
    """Top-level BMKG payload for realtime endpoints."""

    model_config = ConfigDict(extra="allow")

    Infogempa: BMKGInfoGempa