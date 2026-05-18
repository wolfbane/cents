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


# CostCapExceeded is defined above CentsError-derived subclasses to keep the
# hierarchy header docstring accurate; class is intentionally near the top.
