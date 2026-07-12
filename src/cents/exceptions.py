"""Custom exceptions for cents.

Exception Hierarchy:
    CentsError (base)
    ├── ConfigurationError - Missing or invalid configuration
    ├── APIError - External API failures
    │   ├── DataFetchError - Data retrieval failures
    │   └── BrokerError - Broker/trading API failures
    ├── ValidationError - Invalid input data
    └── CostCapExceeded - LLM spend cap (per-run or per-day) would be exceeded
"""


class CentsError(Exception):
    """Base exception for all cents errors."""

    pass


class CostCapExceeded(CentsError):
    """Raised when a cumulative LLM spend cap would be exceeded by the next call.

    The cap is checked PRE-call (using a token estimate) so the offending call
    is never made — preserving the invariant that callers don't pay for work
    that's about to be aborted. Carries both the projected and budgeted spend
    so CLI surfaces can render an actionable message.
    """

    def __init__(
        self,
        message: str,
        *,
        cap_kind: str,  # "run" or "daily"
        cap_usd: float,
        current_usd: float,
        next_call_estimate_usd: float,
    ) -> None:
        self.cap_kind = cap_kind
        self.cap_usd = cap_usd
        self.current_usd = current_usd
        self.next_call_estimate_usd = next_call_estimate_usd
        super().__init__(message)


class ConfigurationError(CentsError):
    """Raised when configuration is missing or invalid."""

    pass


class APIError(CentsError):
    """Raised when an external API call fails."""

    def __init__(self, message: str, service: str = "", status_code: int | None = None):
        self.service = service
        self.status_code = status_code
        super().__init__(message)


class DataFetchError(APIError):
    """Raised when data retrieval from an API fails."""

    pass


class BrokerError(APIError):
    """Raised when broker/trading API operations fail."""

    pass


class ValidationError(CentsError):
    """Raised when input validation fails."""

    pass


class ExperimentConfigDrift(CentsError):
    """Raised when the behavioural-payload SHA differs from the active
    experiment's frozen SHA.

    The SHA covers the *effective* factory config plus adjacent behaviour-
    shifters (prompt templates, model snapshot, EVENT_TAGS) — so a code
    change that adds a FactoryConfig field trips it even when factory.toml
    is byte-identical (that is how pilot_v2 stalled for three weeks).

    Pre-registration discipline: changing behaviour mid-experiment invalidates
    the pre-registered hypothesis. The engine refuses to run unless the operator
    passes --force-frozen-drift to acknowledge the discipline violation. See
    cents-eat0.
    """

    def __init__(
        self,
        experiment_name: str,
        frozen_sha: str,
        current_sha: str,
        drift_detail: list[str] | None = None,
    ):
        self.experiment_name = experiment_name
        self.frozen_sha = frozen_sha
        self.current_sha = current_sha
        self.drift_detail = drift_detail or []
        detail = ""
        if self.drift_detail:
            detail = " Changed: " + "; ".join(self.drift_detail) + "."
        super().__init__(
            f"Behavioural-payload SHA drift detected for active experiment "
            f"{experiment_name!r}: frozen={frozen_sha[:12]} current={current_sha[:12]}. "
            f"The SHA covers the effective factory config + prompts + model "
            f"snapshot + EVENT_TAGS — code changes count, not just factory.toml "
            f"edits.{detail} Mid-experiment changes invalidate pre-registration. "
            f"Pass --force-frozen-drift to override (logs the violation)."
        )


# CostCapExceeded is defined above CentsError-derived subclasses to keep the
# hierarchy header docstring accurate; class is intentionally near the top.
