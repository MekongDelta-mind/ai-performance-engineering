"""Shared benchmark validity-profile constants and helpers."""

from __future__ import annotations

from typing import Final, Tuple

VALIDITY_PROFILE_STRICT: Final[str] = "strict"
VALIDITY_PROFILE_PORTABLE: Final[str] = "portable"
VALIDITY_PROFILE_CHOICES: Final[Tuple[str, str]] = (
    VALIDITY_PROFILE_STRICT,
    VALIDITY_PROFILE_PORTABLE,
)

VALIDITY_PROFILE_HELP_TEXT: Final[str] = (
    "Benchmark validity profile: strict (default; fail-fast with full validity checks) "
    "or portable (compatibility mode for virtualized/limited hosts)."
)

PORTABLE_EXPECTATIONS_UPDATE_HELP_TEXT: Final[str] = (
    "In portable validity profile, expectation writes are disabled by default. "
    "Set this flag to allow expectation-file updates."
)


def normalize_validity_profile(profile: str, *, field_name: str = "validity_profile") -> str:
    """Normalize and validate a validity-profile value."""
    normalized = str(profile).strip().lower()
    if normalized not in VALIDITY_PROFILE_CHOICES:
        allowed = ", ".join(VALIDITY_PROFILE_CHOICES)
        if str(field_name).startswith("--"):
            raise ValueError(
                f"Invalid {field_name} value '{profile}'. Expected one of: {allowed}."
            )
        raise ValueError(f"Invalid {field_name}={profile!r}. Expected one of: {allowed}.")
    return normalized
