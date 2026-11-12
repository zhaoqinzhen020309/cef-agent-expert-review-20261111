"""Keep useful exception causes without credentials."""
def exception_chain(error, secret=None):
    out=[]; seen=set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        message=type(error).__name__+': '+str(error)
        if secret: message=message.replace(secret,'[redacted]')
        out.append(message)
        error=error.__cause__ or error.__context__
    return out
