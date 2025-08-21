# DISABLED - CORE RULE #8 VIOLATION
# This file was importing NautilusTrader packages locally
# NautilusTrader must be used in Docker containers only

from fastapi import HTTPException

def disabled_function(*args, **kwargs):
    raise HTTPException(
        status_code=501,
        detail="File disabled - violates CORE RULE #8. NautilusTrader must be used in Docker containers only."
    )
