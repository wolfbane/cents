"""Custom exceptions for cents.

Exception Hierarchy:
    CentsError (base)
    ├── ConfigurationError - Missing or invalid configuration
    ├── APIError - External API failures
    │   ├── DataFetchError - Data retrieval failures
    │   └── BrokerError - Broker/trading API failures
    └── ValidationError - Invalid input data
"""


class CentsError(Exception):
    """Base exception for all cents errors."""

    pass


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
