"""Network utilities."""


def get_default_host() -> str:
    """Return the default host for connecting to the sandbox.

    Uses localhost since the sandbox server binds to all interfaces.
    """
    return "127.0.0.1"
