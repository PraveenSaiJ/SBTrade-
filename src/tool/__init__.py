import os

import truststore

# Corporate proxy root CA is trusted only by the Windows store; every HTTPS client in this package (Schwab,
# Finnhub) must use it. The Schwab login's local callback server is a SERVER socket in a child process that
# re-imports this package, and truststore breaks server sockets, so login() sets this variable for the child.
if not os.environ.get("SBTRADE_NO_TRUSTSTORE"):
    truststore.inject_into_ssl()
